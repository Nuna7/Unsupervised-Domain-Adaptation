"""
Evaluation metrics for cryo-ET subtomogram segmentation.

Metrics used in the Vox-UDA paper:
    1. mIoU  – mean Intersection over Union (averaged over classes)
    2. Dice  – Dice Similarity Coefficient

Both are computed voxel-wise across the entire volume.

Definitions
-----------
For a single class c, with predicted set P_c and ground-truth set G_c:

    IoU_c  = |P_c ∩ G_c| / |P_c ∪ G_c|
           = TP / (TP + FP + FN)

    Dice_c = 2·|P_c ∩ G_c| / (|P_c| + |G_c|)
           = 2·TP / (2·TP + FP + FN)

mIoU = mean(IoU_c over all classes)
mDice = mean(Dice_c over all classes)

Note: The paper also reports per-class metrics (mIoU_ribo, mIoU_26S, etc.)
which correspond to individual macromolecule categories in the target dataset.
"""

import torch
import numpy as np
from typing import Dict, Optional


class SegmentationMetrics:
    """
    Accumulates predictions and ground-truth over a dataset epoch,
    then computes mIoU and Dice.

    Usage:
        metrics = SegmentationMetrics(num_classes=2)
        for batch in loader:
            pred, gt = model(batch)
            metrics.update(pred, gt)
        results = metrics.compute()

    Args:
        num_classes : number of segmentation classes (2 for binary).
        ignore_index: class index to ignore (-1 means none).
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = -1):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset accumulators for a new evaluation epoch."""
        # Confusion matrix: shape (num_classes, num_classes)
        # conf[i, j] = number of voxels with GT class i predicted as class j
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self,
        preds:  torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Accumulate predictions for one batch.

        Args:
            preds  : (B, D, H, W) long tensor with predicted class indices.
            labels : (B, D, H, W) long tensor with ground-truth class indices.
        """
        preds_np  = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        # Remove ignore-index voxels
        if self.ignore_index >= 0:
            valid     = labels_np != self.ignore_index
            preds_np  = preds_np[valid]
            labels_np = labels_np[valid]

        # Clip to valid class range
        preds_np  = preds_np.clip(0, self.num_classes - 1)
        labels_np = labels_np.clip(0, self.num_classes - 1)

        # Accumulate into confusion matrix
        for gt, pred in zip(labels_np, preds_np):
            self.conf_matrix[gt, pred] += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute mIoU, mDice, and per-class metrics from the accumulated
        confusion matrix.

        Returns:
            dict with keys: 'mIoU', 'mDice', 'IoU_per_class', 'Dice_per_class'
        """
        C   = self.num_classes
        iou = np.zeros(C)
        dsc = np.zeros(C)

        for c in range(C):
            TP = self.conf_matrix[c, c]
            FP = self.conf_matrix[:, c].sum() - TP   # predicted c, but not GT c
            FN = self.conf_matrix[c, :].sum() - TP   # GT c, but not predicted c

            union = TP + FP + FN
            if union > 0:
                iou[c] = TP / union
                dsc[c] = (2.0 * TP) / (2.0 * TP + FP + FN)
            else:
                # Class not present in this split → exclude from average
                iou[c] = float("nan")
                dsc[c] = float("nan")

        mIoU  = np.nanmean(iou)
        mDice = np.nanmean(dsc)

        return {
            "mIoU":          float(mIoU  * 100),
            "mDice":         float(mDice * 100),
            "IoU_per_class": (iou  * 100).tolist(),
            "Dice_per_class":(dsc  * 100).tolist(),
        }

    def __repr__(self) -> str:
        res = self.compute()
        return (
            f"SegmentationMetrics | "
            f"mIoU={res['mIoU']:.2f}%  mDice={res['mDice']:.2f}%"
        )


# ---------------------------------------------------------------------------
# Convenience function for a single-batch metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    preds:       torch.Tensor,
    labels:      torch.Tensor,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute mIoU and Dice for a single batch (no accumulation).

    Args:
        preds      : (B, D, H, W) predicted class indices.
        labels     : (B, D, H, W) ground-truth class indices.
        num_classes: number of classes.

    Returns:
        dict with 'mIoU' and 'mDice' in percentage.
    """
    m = SegmentationMetrics(num_classes=num_classes)
    m.update(preds, labels)
    return m.compute()