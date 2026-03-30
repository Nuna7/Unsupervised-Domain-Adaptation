import os
import sys
import glob
import json
import argparse
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"RUNNING FROM: {os.path.abspath(__file__)}")

from vox_uda                   import VoxUDA
from configs.config            import VoxUDAConfig
from datasets.cryoet_dataset   import (
    SourceDataset, TargetDataset, TargetNoiseSubset,
    build_target_loader, build_eval_loader,
)
from utils.metrics             import SegmentationMetrics

SPLIT_FILE = "data/processed/train_test_split.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("VoxUDA-Train")


# ---------------------------------------------------------------------------
# Source DataLoader that respects train/test split
# ---------------------------------------------------------------------------

def build_source_loader_with_split(
    volume_dir:  str,
    mask_dir:    str,
    batch_size:  int,
    num_workers: int,
    split_file:  str = SPLIT_FILE,
) -> DataLoader:
    """
    Build source DataLoader using only the TRAIN stems from the split file.
    If split file doesn't exist, use all data (with a warning).
    """
    full_dataset = SourceDataset(volume_dir, mask_dir)

    if os.path.exists(split_file):
        with open(split_file) as f:
            split = json.load(f)
        train_stems = set(split["train"])
        test_stems  = set(split["test"])

        # Find indices of training stems in the sorted file list
        all_stems = [
            os.path.splitext(os.path.basename(p))[0]
            for p in full_dataset.volume_paths
        ]
        train_indices = [i for i, s in enumerate(all_stems) if s in train_stems]

        if not train_indices:
            log.warning("No training stems found in dataset — using all data.")
            train_indices = list(range(len(full_dataset)))
        else:
            held_out = [s for s in all_stems if s in test_stems]
            log.info(f"  Split file loaded: using {len(train_indices)} train "
                     f"| holding out {len(held_out)} test samples")

        dataset = Subset(full_dataset, train_indices)
    else:
        log.warning(f"No split file at {split_file} — using ALL {len(full_dataset)} source samples.")
        log.warning("Run: python evaluate.py --build_split  to create a proper evaluation split.")
        dataset = full_dataset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Vox-UDA")
    p.add_argument("--denoiser",         type=str,   default=None)
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--batch_size",       type=int,   default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--ngm_n_sampled",    type=int,   default=None)
    p.add_argument("--ngm_filter_rate",  type=float, default=None)
    p.add_argument("--pseudo_threshold", type=float, default=None)
    p.add_argument("--sigma_d",          type=float, default=None)
    p.add_argument("--sigma_r",          type=float, default=None)
    p.add_argument("--device",           type=str,   default=None)
    p.add_argument("--source_volume_dir",type=str,   default=None)
    p.add_argument("--source_mask_dir",  type=str,   default=None)
    p.add_argument("--target_volume_dir",type=str,   default=None)
    p.add_argument("--checkpoint_dir",   type=str,   default=None)
    p.add_argument("--resume",           type=str,   default=None)
    p.add_argument("--no_split",         action="store_true",
                   help="Ignore split file and use all source data for training.")
    return p.parse_args()


def apply_args(cfg: VoxUDAConfig, args) -> VoxUDAConfig:
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"vox_uda_epoch{epoch:04d}.pt")
    torch.save({
        "epoch":           epoch,
        "student_state":   model.student.state_dict(),
        "teacher_state":   model.teacher.state_dict(),
        "discriminator":   model.discriminator.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config":          cfg.__dict__,
    }, path)
    log.info(f"Checkpoint → {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.student.load_state_dict(ckpt["student_state"])
    model.teacher.load_state_dict(ckpt["teacher_state"])
    model.discriminator.load_state_dict(ckpt["discriminator"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    log.info(f"Resumed from epoch {ckpt['epoch']}  ({path})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Quick source eval during training (using test split if available)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_eval(model, vol_dir, mask_dir, device, max_samples=20, split_file=SPLIT_FILE):
    """
    Fast evaluation on held-out source test split during training.
    Uses at most max_samples to keep it quick.
    """
    net = model.teacher
    net.eval()
    met = SegmentationMetrics(num_classes=2)

    if os.path.exists(split_file):
        with open(split_file) as f:
            stems = json.load(f)["test"]
    else:
        vols  = sorted(glob.glob(os.path.join(vol_dir, "*.npy")))
        stems = [os.path.splitext(os.path.basename(v))[0] for v in vols]

    stems = stems[:max_samples]

    for stem in stems:
        vp = os.path.join(vol_dir,  stem + ".npy")
        mp = os.path.join(mask_dir, stem + ".npy")
        if not os.path.exists(vp) or not os.path.exists(mp):
            continue

        vol  = np.load(vp).astype(np.float32)
        mask = (np.load(mp) > 0).astype(np.int64)
        std  = vol.std()
        if std > 1e-8:
            vol = (vol - vol.mean()) / std

        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        out   = net(vol_t)
        pred  = out["logits"].argmax(dim=1).cpu()
        met.update(pred, torch.from_numpy(mask).unsqueeze(0))

    net.train()
    return met.compute()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: VoxUDAConfig, use_split: bool = True):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────
    if use_split:
        src_loader = build_source_loader_with_split(
            cfg.source_volume_dir, cfg.source_mask_dir,
            cfg.batch_size, cfg.num_workers,
        )
    else:
        from datasets.cryoet_dataset import build_source_loader
        src_loader = build_source_loader(
            cfg.source_volume_dir, cfg.source_mask_dir,
            cfg.batch_size, cfg.num_workers,
        )

    tgt_dataset  = TargetDataset(cfg.target_volume_dir)
    tgt_loader   = build_target_loader(cfg.target_volume_dir, cfg.batch_size, cfg.num_workers)
    noise_subset = TargetNoiseSubset(tgt_dataset, n_sampled=cfg.ngm_n_sampled)

    log.info(f"Source: {len(src_loader.dataset):,}  |  Target: {len(tgt_dataset):,}")

    # ── Model ────────────────────────────────────────────────────────────
    model = VoxUDA(
        in_channels      = cfg.in_channels,
        num_classes      = cfg.num_classes,
        base_channels    = cfg.base_channels,
        denoiser         = cfg.denoiser,
        sigma_d          = cfg.sigma_d,
        sigma_r          = cfg.sigma_r,
        ngm_filter_rate  = cfg.ngm_filter_rate,
        ngm_n_sampled    = cfg.ngm_n_sampled,
        pseudo_threshold = cfg.pseudo_threshold,
        ema_alpha        = cfg.ema_alpha,
        lambda_grl       = cfg.lambda_grl,
        loss_lambdas     = cfg.loss_lambdas,
        loss_alpha       = cfg.loss_alpha,
        loss_beta        = cfg.loss_beta,
        loss_gamma       = cfg.loss_gamma,
    ).to(device)

    trainable = (list(model.student.parameters()) +
                 list(model.discriminator.parameters()))
    optimizer  = optim.Adam(trainable, lr=cfg.lr)
    scheduler  = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay_factor
    )

    start_epoch = 0
    if getattr(cfg, "resume", None):
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, cfg.resume, str(device)
        )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log.info(f"Training {cfg.epochs} epochs  |  denoiser: {cfg.denoiser.upper()}")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, cfg.epochs + 1):

        noise_subset.resample()
        T_Nsampled = noise_subset.as_batch_tensor().to(device)

        tgt_iter = iter(tgt_loader)
        accum    = {k: 0. for k in ["total","l_seg","l_con","l_dis","l_pseudo"]}
        n_batches = 0

        for src_vol, src_mask in src_loader:
            try:
                tgt_vol = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_vol  = next(tgt_iter)

            src_vol  = src_vol.to(device)
            src_mask = src_mask.to(device)
            tgt_vol  = tgt_vol.to(device)

            optimizer.zero_grad()
            loss_d = model.training_step(src_vol, src_mask, tgt_vol, T_Nsampled)
            loss_d["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            model.update_teacher()

            for k in accum:
                accum[k] += loss_d[k].item()
            n_batches += 1

        scheduler.step()

        avg = {k: v / max(n_batches, 1) for k, v in accum.items()}
        log.info(
            f"Epoch {epoch:4d}/{cfg.epochs}  "
            f"total={avg['total']:.4f}  seg={avg['l_seg']:.4f}  "
            f"con={avg['l_con']:.4f}  dis={avg['l_dis']:.4f}  "
            f"pseudo={avg['l_pseudo']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Evaluate on held-out source test split every eval_every epochs
        if epoch % cfg.eval_every == 0:
            res = quick_eval(
                model,
                cfg.source_volume_dir,
                cfg.source_mask_dir,
                str(device),
            )
            log.info(
                f"  [Source test eval]  mIoU={res['mIoU']:.2f}%  "
                f"mDice={res['mDice']:.2f}%"
            )

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg)

    log.info("Training complete.")
    return model


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg  = VoxUDAConfig()
    args = parse_args()
    cfg  = apply_args(cfg, args)
    use_split = not args.no_split
    train(cfg, use_split=use_split)