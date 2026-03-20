"""
Domain Discriminator for Vox-UDA (DANN-style, Gradient Reversal).

Architecture & Output Shape
---------------------------
In the paper, Vox-UDA is described as a *voxel-wise* UDA framework.
Accordingly, the discriminator produces per-voxel domain predictions:

    Input  : fv6  (B, 64, D, H, W)  — deepest student features
    Output : logit (B, 1, D, H, W)  — per-voxel domain score (unnormalised)

This is consistent with voxel-level domain alignment: every spatial location
in the feature map independently votes for source (0) or target (1) domain.
The BCE loss then averages over all B × D × H × W voxels.

Adversarial Mechanism (GRL)
----------------------------
The Gradient Reversal Layer (GRL) is placed between the encoder output
and the discriminator body.

  Forward pass  :  output = input          (identity)
  Backward pass :  ∂L/∂input = −λ · ∂L/∂output  (sign reversal)

Net effect when optimising L_dis = BCE(D(f_v6), domain_label):

  Discriminator ∂   ──  minimises L_dis  →  correct domain classification
  Encoder ∂         ──  maximises L_dis  →  domain-confused features
                         (gradient is negated by GRL before reaching encoder)

λ (lambda_grl) can be annealed from 0 → 1 during training to stabilise
early optimisation (Ganin et al. 2016 schedule), though the paper uses 1.0.

Loss used in vox_uda.py:
    src_domain = discriminator(src_fv6)   # (B, 1, D, H, W)
    tgt_domain = discriminator(tgt_fv6)   # (B, 1, D, H, W)
    L_dis = BCE(src_domain, zeros) + BCE(tgt_domain, ones)
    # BCE averages over all B×D×H×W elements automatically
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GRL(Function):
    """
    Custom autograd: identity forward, negated-and-scaled backward.

    Forward:   y = x
    Backward:  ∂L/∂x = −λ · ∂L/∂y
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scale λ."""
    return _GRL.apply(x, lam)


class GradReversalLayer(nn.Module):
    """Module wrapper around grad_reverse for use inside nn.Sequential."""
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lam)

    def set_lambda(self, lam: float) -> None:
        self.lam = lam


# ---------------------------------------------------------------------------
# Domain Discriminator
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """
    3D voxel-wise domain discriminator.

    Takes fv6 features (B, 64, D, H, W) and returns per-voxel domain logits
    (B, 1, D, H, W) — unnormalised, used with BCEWithLogitsLoss.

    GRL is applied *inside* forward(), so the encoder receives reversed
    gradients automatically whenever this module is called.

    Architecture:
        GRL(λ)
        Conv3d(64→64, 3×3×3, pad=1) + BN + LeakyReLU(0.2)
        Conv3d(64→32, 3×3×3, pad=1) + BN + LeakyReLU(0.2)
        Conv3d(32→1,  1×1×1)
        ─────────────────────────────────────────────────────
        Output: (B, 1, D, H, W)  per-voxel domain logit

    Args:
        in_channels : feature channels from fv6 (default 64).
        lambda_grl  : GRL scale λ (default 1.0; can be annealed externally).
    """

    def __init__(self, in_channels: int = 64, lambda_grl: float = 1.0):
        super().__init__()
        self.grl = GradReversalLayer(lam=lambda_grl)

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 1, kernel_size=1),        # (B, 1, D, H, W)
        )

        # Kaiming init for conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (B, 64, D, H, W) fv6 features from student encoder.
        Returns:
            logit    : (B, 1, D, H, W)  per-voxel domain score.
                       BCEWithLogitsLoss averages over all spatial positions.
        """
        x = self.grl(features)
        return self.net(x)   # (B, 1, D, H, W)