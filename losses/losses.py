"""
Loss Functions for Vox-UDA.

Total objective:
    L = L_seg + α·L_con + β·L_dis + γ·L_pseudo

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  Segmentation Loss  L_seg = L_CE + L_Dice
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft multiclass Dice loss.

        L_Dice = 1 − mean_c [ (2·|P_c ∩ G_c| + ε) / (|P_c| + |G_c| + ε) ]

    Args:
        smooth       : Laplace smoothing ε (default 1.0).
        ignore_index : voxel label to exclude (default -1).
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, D, H, W) raw network output.
            targets : (B, D, H, W) long tensor, may contain ignore_index values.
        Returns:
            scalar Dice loss.
        """
        C = logits.shape[1]

        # --- Valid mask: True where the label is not ignored ---
        valid = (targets != self.ignore_index)      # (B, D, H, W)

        # --- Clamp targets so F.one_hot doesn't see -1 ---
        targets_safe = targets.clamp(min=0)          # (B, D, H, W)

        probs     = F.softmax(logits, dim=1)          # (B, C, D, H, W)

        # One-hot encode (B, D, H, W) → (B, D, H, W, C) → (B, C, D, H, W)
        target_oh = F.one_hot(targets_safe, num_classes=C)     # (B,D,H,W,C)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()  # (B,C,D,H,W)

        # Zero out ignored voxels in BOTH probs and one-hot targets
        valid_c = valid.unsqueeze(1).float()         # (B, 1, D, H, W)
        probs     = probs     * valid_c
        target_oh = target_oh * valid_c

        # Flatten spatial dims
        probs_f  = probs.reshape(probs.shape[0],     C, -1)   # (B, C, N)
        target_f = target_oh.reshape(target_oh.shape[0], C, -1)

        intersection = (probs_f * target_f).sum(dim=-1)       # (B, C)
        union        = probs_f.sum(dim=-1) + target_f.sum(dim=-1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    L_seg = L_CE + L_Dice  (on SOURCE domain with GROUND-TRUTH masks).

    Args:
        ce_weight    : weight on CE term   (default 1.0).
        dice_weight  : weight on Dice term (default 1.0).
        ignore_index : label to ignore in both CE and Dice (default -1).
    """

    def __init__(
        self,
        ce_weight:    float = 1.0,
        dice_weight:  float = 1.0,
        ignore_index: int   = -1,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce_fn   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_fn = DiceLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C, D, H, W)
        targets: torch.Tensor,   # (B, D, H, W) long — ground-truth, no -1
    ) -> torch.Tensor:
        ce_loss   = self.ce_fn(logits, targets)
        dice_loss = self.dice_fn(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# 2.  Consistency Loss  L_con
# ---------------------------------------------------------------------------

def cosine_loss(f: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    L_BN(f, f') = 1 − mean_{voxels} cos_sim(f_v, f'_v)

    cos_sim is computed along the CHANNEL dimension at each voxel position.

    Args:
        f, fp : (B, C, D, H, W)
    Returns:
        scalar in [0, 2].
    """
    B, C = f.shape[:2]
    f_flat  = f.view(B, C, -1)    # (B, C, N)
    fp_flat = fp.view(B, C, -1)
    sim = F.cosine_similarity(f_flat, fp_flat, dim=1)   # (B, N)
    return 1.0 - sim.mean()


class ConsistencyLoss(nn.Module):
    """
    L_con = Σ_k  λ_k · L_BN(f_k, f'_k)

    Applied to all four feature levels (fc, fv2, fv4, fv6).
    Weights reflect semantic importance:
        shallow (texture) → lower λ,  deep (edges) → higher λ.

    Default from paper ablation: [0.2, 0.2, 0.3, 0.3]
    """

    KEYS = ["fc", "fv2", "fv4", "fv6"]

    def __init__(self, lambdas: list = None):
        super().__init__()
        self.lambdas = lambdas if lambdas is not None else [0.2, 0.2, 0.3, 0.3]
        assert len(self.lambdas) == 4

    def forward(self, feats_clean: dict, feats_noisy: dict) -> torch.Tensor:
        """
        Args:
            feats_clean : {fc, fv2, fv4, fv6} from clean source x^s
            feats_noisy : {fc, fv2, fv4, fv6} from noisy source x^{s'}
        Returns:
            scalar consistency loss.
        """
        # Accumulate by starting from first term (avoids leaf-tensor issue)
        total = None
        for lam, key in zip(self.lambdas, self.KEYS):
            term = lam * cosine_loss(feats_clean[key], feats_noisy[key])
            total = term if total is None else total + term
        return total


# ---------------------------------------------------------------------------
# 3.  Discriminator Loss  L_dis  (voxel-wise BCE)
# ---------------------------------------------------------------------------

class DiscriminatorLoss(nn.Module):
    """
    L_dis = BCE(D(fv6_src), 0) + BCE(D(fv6_tgt), 1)

    Discriminator output is (B, 1, D, H, W) — per-voxel domain logits.
    BCEWithLogitsLoss computes element-wise BCE and then averages over
    ALL elements (B × 1 × D × H × W), which is equivalent to averaging
    the per-voxel domain predictions.

    Source domain label = 0  (simulated)
    Target domain label = 1  (experimental)
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()   # handles any shape

    def forward(
        self,
        domain_logits: torch.Tensor,   # (B, 1, D, H, W) or (B, 1)
        is_target: bool,
    ) -> torch.Tensor:
        label = torch.ones_like(domain_logits) if is_target \
                else torch.zeros_like(domain_logits)
        return self.bce(domain_logits, label)


# ---------------------------------------------------------------------------
# 4.  Pseudo-label Loss  L_pseudo  (target domain, CE only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_pseudo_labels(
    teacher_logits: torch.Tensor,
    threshold:      float = 0.85,
) -> tuple:
    """
    Hard pseudo-labels with confidence gating.

        p_v     = softmax(logit_v)
        ỹ_v     = argmax_c p_v          (predicted class)
        mask_v  = (max_c p_v ≥ η)      (confidence gate)

    Args:
        teacher_logits : (B, C, D, H, W) from teacher on denoised target.
        threshold      : η = 0.85 (paper default).

    Returns:
        pseudo_labels   : (B, D, H, W) long — class indices (0 or 1).
        confidence_mask : (B, D, H, W) bool — True at accepted voxels.
    """
    probs               = F.softmax(teacher_logits, dim=1)
    confidence, labels  = probs.max(dim=1)       # (B, D, H, W) each
    mask                = confidence >= threshold
    return labels, mask


class PseudoLabelLoss(nn.Module):
    """
    L_pseudo = CE(student(x^t), ỹ^t)  [only at voxels where conf ≥ η]

    Args:
        threshold    : η, teacher confidence gate (default 0.85).
        ignore_index : label value to skip in CE (default -1).
    """

    def __init__(self, threshold: float = 0.85, ignore_index: int = -1):
        super().__init__()
        self.threshold    = threshold
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(
        self,
        student_logits: torch.Tensor,    # (B, C, D, H, W) student on target
        teacher_logits: torch.Tensor,    # (B, C, D, H, W) teacher on denoised
    ) -> tuple:
        """
        Returns:
            loss          : scalar CE loss (or 0 if no high-confidence voxels)
            pseudo_labels : (B, D, H, W) predicted labels from teacher
            mask          : (B, D, H, W) bool — accepted voxels
        """
        pseudo_labels, mask = generate_pseudo_labels(teacher_logits, self.threshold)

        # Set low-confidence voxels to ignore_index
        masked_labels          = pseudo_labels.clone()
        masked_labels[~mask]   = self.ignore_index

        loss = self.ce_fn(student_logits, masked_labels)
        return loss, pseudo_labels, mask


# ---------------------------------------------------------------------------
# 5.  Combined Vox-UDA Loss Wrapper
# ---------------------------------------------------------------------------

class VoxUDALoss(nn.Module):
    """
    L = L_seg + α·L_con + β·L_dis + γ·L_pseudo

    Args:
        lambdas       : [λ₁,λ₂,λ₃,λ₄] weights for L_con per feature level.
        alpha         : weight for L_con (noise robustness term).
        beta          : weight for L_dis (adversarial domain alignment term).
        gamma         : weight for L_pseudo (target supervision term).
        pseudo_thresh : teacher confidence threshold η for pseudo-labels.
    """

    def __init__(
        self,
        lambdas:       list  = None,
        alpha:         float = 1.0,
        beta:          float = 1.0,
        gamma:         float = 1.0,
        pseudo_thresh: float = 0.85,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.seg_loss    = SegmentationLoss()
        self.con_loss    = ConsistencyLoss(lambdas=lambdas)
        self.dis_loss    = DiscriminatorLoss()
        self.pseudo_loss = PseudoLabelLoss(threshold=pseudo_thresh)

    def forward(
        self,
        # ── Source / ground-truth supervision ──
        src_logits:        torch.Tensor,   # (B, C, D, H, W)
        src_labels:        torch.Tensor,   # (B, D, H, W) long, no -1
        # ── Feature consistency (clean vs. noisy source) ──
        src_feats:         dict,           # {fc, fv2, fv4, fv6}
        src_noisy_feats:   dict,           # {fc, fv2, fv4, fv6}
        # ── Domain discrimination ──
        src_domain_logits: torch.Tensor,   # (B, 1, D, H, W) discriminator output
        tgt_domain_logits: torch.Tensor,   # (B, 1, D, H, W)
        # ── Target pseudo-label supervision ──
        tgt_student_logits: torch.Tensor,  # (B, C, D, H, W)
        tgt_teacher_logits: torch.Tensor,  # (B, C, D, H, W)
    ) -> dict:

        l_seg = self.seg_loss(src_logits, src_labels)

        l_con = self.con_loss(src_feats, src_noisy_feats)

        l_dis = (self.dis_loss(src_domain_logits, is_target=False)
               + self.dis_loss(tgt_domain_logits, is_target=True))

        l_pseudo, pseudo_labels, pseudo_mask = \
            self.pseudo_loss(tgt_student_logits, tgt_teacher_logits)

        total = (l_seg
               + self.alpha * l_con
               + self.beta  * l_dis
               + self.gamma * l_pseudo)

        return {
            "total":         total,
            "l_seg":         l_seg,
            "l_con":         l_con,
            "l_dis":         l_dis,
            "l_pseudo":      l_pseudo,
            "pseudo_labels": pseudo_labels,
            "pseudo_mask":   pseudo_mask,
        }