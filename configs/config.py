"""
Hyperparameter configuration for Vox-UDA training.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class VoxUDAConfig:
    BASE = "/shared/scratch/0/home/v_zonunmawia_zadeng/UDA/"
    # ---- Model ----
    in_channels:       int   = 1
    num_classes:       int   = 2
    base_channels:     int   = 32

    # ---- Denoiser choice: 'ibf' | 'bf' | 'ngm' ---- # I just add NGM as denoiser just to consider.
    denoiser:          str   = "ibf"

    # ---- IBF / BF hyperparameters ----
    sigma_d:           float = 120.0    # domain (spatial) bandwidth
    sigma_r:           float = 1.2     # range (gradient/intensity) bandwidth
    window:            int   = 3       # sliding window size

    # ---- NGM hyperparameters ----
    ngm_filter_rate:   float = 0.244   # high-pass filter keep-rate ρ (24.4%)
    ngm_n_sampled:     int   = 10      # N_sampled

    # ---- Pseudo-labeling ----
    pseudo_threshold:  float = 0.85    # teacher confidence threshold η

    # ---- EMA ----
    ema_alpha:         float = 0.999   # teacher EMA decay

    # ---- Discriminator ----
    lambda_grl:        float = 1.0     # Gradient Reversal Layer scaling

    # ---- Loss weights ----
    # Consistency loss layer weights [λ₁, λ₂, λ₃, λ₄]
    # Paper ablation shows [0.2, 0.2, 0.3, 0.3] is best:
    #   shallow (fc, fv2) → texture → lower weight
    #   deep    (fv4, fv6) → edges  → higher weight
    loss_lambdas:      List[float] = field(default_factory=lambda: [0.2, 0.2, 0.3, 0.3])
    loss_alpha:        float = 1.0     # weight for L_con
    loss_beta:         float = 1.0     # weight for L_dis
    loss_gamma:        float = 1.0     # weight for L_pseudo

    # ---- Training ----
    epochs:            int   = 300
    batch_size:        int   = 16
    lr:                float = 1e-3    # initial learning rate
    lr_decay_factor:   float = 0.1    # decay LR by 90% every lr_step epochs
    lr_step:           int   = 100    # epoch interval for LR decay
    optimizer:         str   = "adam"

    # ---- Hardware ----
    device:            str   = "cuda"
    num_workers:       int   = 0

    # ---- Evaluation ----
    eval_every:        int   = 10     # evaluate on target val set every N epochs
    save_every:        int   = 50     # save checkpoint every N epochs

    # ---- Paths ----
    source_volume_dir: str   = BASE + "data/source/volumes"
    source_mask_dir:   str   = BASE + "data/source/masks"
    target_volume_dir: str   = BASE + "data/target/volumes"
    target_eval_dir:   str   = BASE + "data/target/eval"    # for evaluation only
    target_mask_dir:   str   = BASE + "data/target/masks"   # eval masks
    checkpoint_dir:    str   = BASE + "checkpoints"
    log_dir:           str   = BASE + "logs"