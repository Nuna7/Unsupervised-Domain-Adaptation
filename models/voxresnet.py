"""
VoxResNet: 3D Voxelwise Residual Network (backbone for both student and teacher).

Architecture — matches Figure 2 exactly
-----------------------------------------
Figure 2 shows the following sequence of blocks (left → right):

    Conv 3x3x3, 32 | BatchNorm | ReLU |
    Conv 3x3x3, 32 | BatchNorm | ReLU |    ← fc tap (L_con #1, λ₁)
    Conv 3x3x3, 64 | VoxReS   | VoxReS |  ← fv2 tap (L_con #2, λ₂)
    BatchNorm | ReLU |
    Conv 3x3x3, 64 | VoxReS   | VoxReS |  ← fv4 tap (L_con #3, λ₃)
    BatchNorm | ReLU |
    Conv 3x3x3, 64 | VoxReS   | VoxReS |  ← fv6 tap (L_con #4, λ₄) + L_dis

L_seg comes from the final segmentation head (source flow, black arrow).
L_dis comes from fv6 features fed to the domain discriminator (red arrow).
L_con taps appear after each of the four named feature levels.
Pseudo-labels (purple arrow) come from the teacher's forward pass.
EMA (orange arrow down) updates teacher weights from student weights.

VoxRes Block
------------
    ┌─ Conv3d(C→C, 3×3×3) ─ BN ─ ReLU ─ Conv3d(C→C, 3×3×3) ─ BN ─┐
    │                                                                   │
    x ────────────────────── shortcut ─────────────────────────────── + ─ ReLU ─ out

If C_in ≠ C_out, the shortcut uses a 1×1×1 projection conv.

Forward returns a dict so all feature taps are accessible to VoxUDA:
    {
        'logits': (B, num_classes, D, H, W),
        'fc'    : (B, 32, D, H, W),
        'fv2'   : (B, 64, D, H, W),
        'fv4'   : (B, 64, D, H, W),
        'fv6'   : (B, 64, D, H, W),
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxResBlock(nn.Module):
    """
    3D residual block with optional channel-projection shortcut.

    Forward:
        identity = shortcut(x)          [1×1×1 conv if C_in ≠ C_out, else identity]
        z = ReLU(BN(Conv(x)))
        z = BN(Conv(z))
        return ReLU(z + identity)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,  out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)

        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)),  inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class VoxResNet(nn.Module):
    """
    Full VoxResNet producing segmentation logits + four intermediate feature maps.

    Args:
        in_channels  : input channels (1 for greyscale cryo-ET).
        num_classes  : output classes (2 for binary segmentation).
        base_channels: initial width C (default 32; doubled to 64 at vg1).
    """

    def __init__(
        self,
        in_channels:   int = 1,
        num_classes:   int = 2,
        base_channels: int = 32,
    ):
        super().__init__()
        C = base_channels      # 32
        D = base_channels * 2  # 64

        # ── fc: two plain conv layers ──────────────────────────────────────
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm3d(C), nn.ReLU(inplace=True),
            nn.Conv3d(C, C,          3, padding=1, bias=False),
            nn.BatchNorm3d(C), nn.ReLU(inplace=True),
        )

        # ── fv2: Conv(C→D) + BN + ReLU + 2×VoxRes(D) ──────────────────────
        self.vg1_proj = nn.Conv3d(C, D, 3, padding=1, bias=False)
        self.vg1_bn   = nn.BatchNorm3d(D)
        self.vg1_res  = nn.Sequential(VoxResBlock(D, D), VoxResBlock(D, D))

        # ── fv4: Conv(D→D) + BN + ReLU + 2×VoxRes(D) ──────────────────────
        self.vg2_proj = nn.Conv3d(D, D, 3, padding=1, bias=False)
        self.vg2_bn   = nn.BatchNorm3d(D)
        self.vg2_res  = nn.Sequential(VoxResBlock(D, D), VoxResBlock(D, D))

        # ── fv6: Conv(D→D) + BN + ReLU + 2×VoxRes(D) ──────────────────────
        self.vg3_proj = nn.Conv3d(D, D, 3, padding=1, bias=False)
        self.vg3_bn   = nn.BatchNorm3d(D)
        self.vg3_res  = nn.Sequential(VoxResBlock(D, D), VoxResBlock(D, D))

        # ── Segmentation head: multi-scale fusion ──────────────────────────
        # Concatenate all four feature levels: (C + D + D + D) = 32+64+64+64 = 224
        fused_ch = C + D + D + D
        self.fuse = nn.Sequential(
            nn.Conv3d(fused_ch, D, 1, bias=False),
            nn.BatchNorm3d(D),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv3d(D, num_classes, 1)

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : (B, 1, D, H, W) greyscale subtomogram.
        Returns:
            dict with keys: logits, fc, fv2, fv4, fv6.
        """
        # fc  — shallow texture features
        fc  = self.init_conv(x)

        # fv2 — mid-level features
        fv2 = F.relu(self.vg1_bn(self.vg1_proj(fc)), inplace=True)
        fv2 = self.vg1_res(fv2)

        # fv4 — deeper structural features
        fv4 = F.relu(self.vg2_bn(self.vg2_proj(fv2)), inplace=True)
        fv4 = self.vg2_res(fv4)

        # fv6 — deepest edge/boundary features + discriminator input
        fv6 = F.relu(self.vg3_bn(self.vg3_proj(fv4)), inplace=True)
        fv6 = self.vg3_res(fv6)

        # Multi-scale fusion → segmentation logits
        fused  = torch.cat([fc, fv2, fv4, fv6], dim=1)   # (B, 224, D, H, W)
        fused  = self.fuse(fused)
        logits = self.seg_head(fused)                      # (B, num_classes, D, H, W)

        return {"logits": logits, "fc": fc, "fv2": fv2, "fv4": fv4, "fv6": fv6}

    @torch.no_grad()
    def copy_weights_from(self, other: "VoxResNet") -> None:
        """Hard-copy weights from another VoxResNet (used to initialise teacher)."""
        self.load_state_dict(other.state_dict())


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_teacher_ema(
    student: nn.Module,
    teacher: nn.Module,
    alpha:   float = 0.999,
) -> None:
    """
    EMA update:  θ_teacher ← α·θ_teacher + (1−α)·θ_student

    Must be called AFTER optimizer.step().
    Teacher parameters are never updated by back-propagation — only by EMA.

    Args:
        student : student network (gradient-trained).
        teacher : teacher network (EMA-updated, no grad).
        alpha   : decay factor (0.999 from FixMatch / Mean Teacher).
    """
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(alpha).add_(p_s.data, alpha=1.0 - alpha)