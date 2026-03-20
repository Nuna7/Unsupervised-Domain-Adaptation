"""
Vox-UDA: Full Framework Integration.

This module wires together all components into a single nn.Module:

    student      ← VoxResNet, trained by Adam on all 4 losses
    teacher      ← VoxResNet, updated ONLY by EMA; provides pseudo-labels
    discriminator← 3D convnet with GRL; produces per-voxel domain logits
    ngm          ← Noise Generation Module; injects target-like Gaussian noise
    denoiser     ← IBF / BF / NGM; cleans target before teacher sees it

Three data flows (matching Figure 2 colors):

    Black  (Source flow):
        x^s → Student → L_seg (supervised with GT masks y^s)
        x^s → Student → features {fc, fv2, fv4, fv6} → L_con term 1

    Blue   (Noisy flow):
        x^{s'} = x^s + ε_NGM → Student → features → L_con term 2
        L_con = Σ_k λ_k · cosine_similarity(f_k^clean, f_k^noisy)

    Red    (Target flow):
        x^t → Student → fv6 → Discriminator → L_dis
        IBF(x^t) → Teacher → ỹ^t (pseudo-labels, conf ≥ η)
        ỹ^t + Student(x^t) → L_pseudo

    Orange arrow (EMA): θ_teacher ← 0.999·θ_teacher + 0.001·θ_student
    Purple arrow (pseudo-label): teacher output passed back to student loss

Discriminator output shape
--------------------------
The discriminator returns (B, 1, D, H, W) — per-voxel domain logits.
BCEWithLogitsLoss averages over all B × D × H × W positions, which
gives a stable voxel-wise domain alignment signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.voxresnet     import VoxResNet, update_teacher_ema
from models.discriminator import DomainDiscriminator
from modules.ngm          import NoiseGenerationModule
from modules.ibf          import build_denoiser
from losses.losses        import VoxUDALoss


class VoxUDA(nn.Module):
    """
    Full Vox-UDA framework.

    Args:
        in_channels      : input channels (1 for greyscale cryo-ET).
        num_classes      : segmentation classes (2 = binary).
        base_channels    : VoxResNet base width (32, paper default).
        denoiser         : 'ibf' | 'bf' | 'ngm' — used before teacher.
        sigma_d          : IBF/BF domain (spatial) bandwidth (120).
        sigma_r          : IBF/BF range (gradient) bandwidth (1.2).
        ngm_filter_rate  : NGM high-pass keep-rate ρ (0.244 = 24.4%).
        ngm_n_sampled    : target samples for NGM per batch (10).
        pseudo_threshold : teacher confidence gate η (0.85).
        ema_alpha        : teacher EMA decay (0.999).
        lambda_grl       : GRL scaling factor (1.0).
        loss_lambdas     : [λ₁,λ₂,λ₃,λ₄] for L_con layers.
        loss_alpha/beta/gamma : weights for L_con, L_dis, L_pseudo.
    """

    def __init__(
        self,
        in_channels:      int   = 1,
        num_classes:      int   = 2,
        base_channels:    int   = 32,
        denoiser:         str   = "ibf",
        sigma_d:          float = 120.0,
        sigma_r:          float = 1.2,
        ngm_filter_rate:  float = 0.244,
        ngm_n_sampled:    int   = 10,
        pseudo_threshold: float = 0.85,
        ema_alpha:        float = 0.999,
        lambda_grl:       float = 1.0,
        loss_lambdas:     list  = None,
        loss_alpha:       float = 1.0,
        loss_beta:        float = 1.0,
        loss_gamma:       float = 1.0,
    ):
        super().__init__()

        # ── Student (gradient-trained) ──────────────────────────────────
        self.student = VoxResNet(in_channels, num_classes, base_channels)

        # ── Teacher (EMA-only, no gradient) ────────────────────────────
        self.teacher = VoxResNet(in_channels, num_classes, base_channels)
        self.teacher.copy_weights_from(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ── Domain discriminator ────────────────────────────────────────
        # Input: fv6  channels = base_channels × 2 = 64
        # Output: (B, 1, D, H, W) per-voxel domain logits
        self.discriminator = DomainDiscriminator(
            in_channels=base_channels * 2,
            lambda_grl=lambda_grl,
        )

        # ── NGM ─────────────────────────────────────────────────────────
        self.ngm           = NoiseGenerationModule(ngm_filter_rate, ngm_n_sampled)
        self.denoiser_type = denoiser

        # ── IBF / BF denoiser ───────────────────────────────────────────
        if denoiser in ("ibf", "bf"):
            self.denoiser = build_denoiser(denoiser, sigma_d=sigma_d, sigma_r=sigma_r)
        else:
            self.denoiser = None   # NGM-based denoising handled inline

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = VoxUDALoss(
            lambdas       = loss_lambdas,
            alpha         = loss_alpha,
            beta          = loss_beta,
            gamma         = loss_gamma,
            pseudo_thresh = pseudo_threshold,
        )
        self.ema_alpha = ema_alpha

    # -----------------------------------------------------------------------
    # Denoising dispatch
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def denoise_target(self, target: torch.Tensor) -> torch.Tensor:
        """Apply configured denoiser to target volumes before teacher pass."""
        if self.denoiser_type == "ngm":
            return self.ngm.denoise(target)
        return self.denoiser(target)

    # -----------------------------------------------------------------------
    # Training step  (one batch)
    # -----------------------------------------------------------------------

    def training_step(
        self,
        source:        torch.Tensor,   # (B, 1, D, H, W)  labelled source
        source_labels: torch.Tensor,   # (B, D, H, W)     binary GT masks
        target:        torch.Tensor,   # (B, 1, D, H, W)  unlabelled target
        target_subset: torch.Tensor,   # (N, 1, D, H, W)  NGM noise subset
    ) -> dict:
        """
        One forward pass.  Returns dict with:
          'total'  — differentiable total loss (call .backward() on this)
          'l_seg', 'l_con', 'l_dis', 'l_pseudo' — individual terms (.item())
          'pseudo_labels', 'pseudo_mask'         — teacher predictions
        """

        # ── 1. NGM: add target-like noise to source ──────────────────────
        # Per-sample variance → per-sample Gaussian → average (Figure 2)
        ngm_out      = self.ngm(source, target_subset)
        noisy_source = ngm_out["noisy_source"]           # x^{s'} = x^s + ε_avg

        # ── 2. Student forward passes ────────────────────────────────────

        # (a) Clean source  →  logits_s, features for L_seg and L_con
        src_out   = self.student(source)        # {logits, fc, fv2, fv4, fv6}

        # (b) Noisy source  →  features for L_con (the "noisy" side)
        noisy_out = self.student(noisy_source)  # {logits, fc, fv2, fv4, fv6}

        # (c) Target        →  logits for L_pseudo, fv6 for L_dis
        tgt_out   = self.student(target)        # {logits, fc, fv2, fv4, fv6}

        # ── 3. Discriminator: per-voxel domain logits ────────────────────
        # Output shape: (B, 1, D, H, W)
        src_domain = self.discriminator(src_out["fv6"])   # source domain
        tgt_domain = self.discriminator(tgt_out["fv6"])   # target domain

        # ── 4. Teacher on DENOISED target → pseudo-labels ───────────────
        with torch.no_grad():
            denoised_target = self.denoise_target(target)
            teacher_out     = self.teacher(denoised_target)
            teacher_logits  = teacher_out["logits"]       # (B, C, D, H, W)

        # ── 5. Compute all losses ────────────────────────────────────────
        loss_dict = self.criterion(
            src_logits         = src_out["logits"],
            src_labels         = source_labels,
            src_feats          = {k: src_out[k]   for k in ["fc","fv2","fv4","fv6"]},
            src_noisy_feats    = {k: noisy_out[k] for k in ["fc","fv2","fv4","fv6"]},
            src_domain_logits  = src_domain,
            tgt_domain_logits  = tgt_domain,
            tgt_student_logits = tgt_out["logits"],
            tgt_teacher_logits = teacher_logits,
        )

        return loss_dict

    # -----------------------------------------------------------------------
    # EMA teacher update  (call AFTER optimizer.step())
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self) -> None:
        """θ_teacher ← 0.999·θ_teacher + 0.001·θ_student"""
        update_teacher_ema(self.student, self.teacher, alpha=self.ema_alpha)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        volume:      torch.Tensor,
        use_teacher: bool = True,
    ) -> torch.Tensor:
        """
        Voxel-wise segmentation prediction.

        Args:
            volume      : (B, 1, D, H, W)
            use_teacher : True (recommended) to use the EMA teacher.
        Returns:
            pred : (B, D, H, W) long tensor with class indices.
        """
        net = self.teacher if use_teacher else self.student
        net.eval()
        out = net(volume)
        net.train()
        return out["logits"].argmax(dim=1)

    @torch.no_grad()
    def predict_proba(
        self,
        volume:      torch.Tensor,
        use_teacher: bool = True,
    ) -> torch.Tensor:
        """
        Returns softmax class probabilities (B, C, D, H, W).
        """
        net = self.teacher if use_teacher else self.student
        net.eval()
        out = net(volume)
        net.train()
        return F.softmax(out["logits"], dim=1)