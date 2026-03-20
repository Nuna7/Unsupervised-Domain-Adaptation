# Vox-UDA: Technical Implementation Documentation

**Paper:** "Vox-UDA: Voxel-wise Unsupervised Domain Adaptation for Cryo-Electron Subtomogram Segmentation with Denoised Pseudo-Labeling"  

# Note: The code base is self-implemented and currently only include the architecture implementation,so when data is available the training code will be put. The explanation and documentation is also self-written based on personal understanding.

---

## Table of Contents

1. [Problem Context and Motivation](#1-problem-context-and-motivation)
2. [Framework Overview](#2-framework-overview)
3. [VoxResNet: The Segmentation Backbone](#3-voxresnet-the-segmentation-backbone)
4. [Noise Generation Module (NGM)](#4-noise-generation-module-ngm)
5. [Denoised Pseudo-Labeling (DPL): BF → IBF](#5-denoised-pseudo-labeling-dpl-bf--ibf)
6. [Student–Teacher EMA Framework](#6-studentteacher-ema-framework)
7. [Loss Functions: Full Derivation](#7-loss-functions-full-derivation)
8. [Domain Discriminator and Adversarial Training](#8-domain-discriminator-and-adversarial-training)
9. [Training Flow: Step-by-Step](#9-training-flow-step-by-step)
10. [Hyperparameter Guide](#10-hyperparameter-guide)
11. [Repository Structure](#11-repository-structure)
12. [Usage Guide](#12-usage-guide)

---

## 1. Problem Context and Motivation

### What is Cryo-ET?

Cryo-Electron Tomography (cryo-ET) is a 3D imaging modality that captures macromolecular structures inside biological cells at near-atomic resolution. A **tomogram** is a full 3D reconstruction of a cell section; a **subtomogram** is a small cubic sub-volume (typically 32×32×32 voxels) that contains a single macromolecular complex. **Subtomogram segmentation** is the task of labelling each voxel as foreground (macromolecule) or background.

### The Two Core Challenges

**Challenge 1 – Cross-noise-level gap:**  
Simulated (source) subtomograms are generated with *fixed, known* SNR levels (0.03 or 0.05 dB). Real experimental (target) subtomograms collected from actual biological specimens have *unpredictable, variable* noise levels depending on equipment, sample thickness, and radiation dose. A model trained only on simulated data generalises poorly to real data because it has never encountered the noise distribution it will face at test time.

**Challenge 2 – Domain shift from unknown macromolecules:**  
The source dataset contains known macromolecule classes (e.g., ribosomes, 26S proteasome, TRiC). Target datasets may contain *different or completely unknown* macromolecules. The model trained to recognise source molecules will be biased toward their structural signatures, leading to poor segmentation of target molecules.

### The Unsupervised Domain Adaptation (UDA) Setting

```
Source domain S = {(x^s_i, y^s_i)}^N_{i=1}    ← labelled (simulated)
Target domain T = {x^t_j}^M_{j=1}             ← UNLABELLED (experimental)
```

The goal is to train a segmentation network that performs well on T, using only:
- Full voxel-level supervision from S (ground-truth masks y^s)
- Unlabelled volumes from T (no ground truth at training time)

---

## 2. Framework Overview

Vox-UDA consists of two modules operating in tandem:

```
┌─────────────────────────────────────────────────────────────────┐
│                        VOX-UDA FRAMEWORK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐    ┌─────────────────────────────────────┐ │
│  │  SOURCE FLOW     │    │  TARGET FLOW                        │ │
│  │                  │    │                                     │ │
│  │  x^s ──────────► Student ──► L_seg (ground-truth masks)    │ │
│  │       │          │    │                                     │ │
│  │  NGM  │          │    │  x^t ──────────► Student ──► L_dis │ │
│  │  (ε)  ▼          │    │                                     │ │
│  │  x^s' ─────────► Student ──► L_con (vs. clean features)    │ │
│  │                  │    │                                     │ │
│  └──────────────────┘    │  IBF(x^t) ──► Teacher ──► ỹ^t      │ │
│                          │                    │                  │ │
│                          │              ỹ^t ──► Student ──► L_pseudo │
│                          └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Total Loss:**
```
L = L_seg + L_con + L_dis + L_pseudo
```

---

## 3. VoxResNet: The Segmentation Backbone

### Architecture

VoxResNet (Chen et al., 2018) is a fully 3D residual network designed for voxel-level segmentation. In Vox-UDA, an *identical architecture* is used for both the **student** and **teacher** networks.

```
Input: (B, 1, 32, 32, 32)  — single-channel greyscale subtomogram
│
├─ Conv3d(1→32, 3×3×3) + BN + ReLU
├─ Conv3d(32→32, 3×3×3) + BN + ReLU
│                                        → fc  (B, 32, D, H, W)  [texture features]
├─ Conv3d(32→64, 3×3×3) + BN + ReLU
├─ VoxResBlock(64, 64)
├─ VoxResBlock(64, 64)
│                                        → fv2 (B, 64, D, H, W)  [mid-level features]
├─ Conv3d(64→64, 3×3×3) + BN + ReLU
├─ VoxResBlock(64, 64)
├─ VoxResBlock(64, 64)
│                                        → fv4 (B, 64, D, H, W)  [deeper features]
├─ Conv3d(64→64, 3×3×3) + BN + ReLU
├─ VoxResBlock(64, 64)
├─ VoxResBlock(64, 64)
│                                        → fv6 (B, 64, D, H, W)  [edge/structure features]
│
├─ Multi-scale fusion: concat(fc, fv2, fv4, fv6) → Conv1×1×1(224→64) + BN + ReLU
└─ Segmentation head: Conv1×1×1(64→2)
   Output: (B, 2, 32, 32, 32)  — per-voxel class logits
```

### VoxRes Block

Each VoxRes block is a 3D residual block:

```
input x
  │
  ├─► Conv3d(C→C, 3×3×3) → BN → ReLU
  │         ↓
  │   Conv3d(C→C, 3×3×3) → BN
  │         ↓
  └─ shortcut(x) ──────────── +
                               ↓
                              ReLU → output
```

The residual connection allows gradients to flow directly through the network, enabling training of deep 3D networks without vanishing gradients. If input channels ≠ output channels, a 1×1×1 projection convolution adapts the shortcut.

### Why these four feature levels?

| Level | Channels | Depth | Content             | Role in L_con |
|-------|----------|-------|---------------------|---------------|
| fc    | 32       | 2     | Low-level texture   | Weight λ₁=0.2 |
| fv2   | 64       | 4     | Mid-level patterns  | Weight λ₂=0.2 |
| fv4   | 64       | 6     | Structural shapes   | Weight λ₃=0.3 |
| fv6   | 64       | 8     | Edges, boundaries   | Weight λ₄=0.3 |

Shallower layers encode fine-grained texture (easier to match across noise levels), while deeper layers encode structural information. The consistency loss weights these accordingly.

---

## 4. Noise Generation Module (NGM)

### Motivation

The model trained only on fixed-noise source data learns to segment despite a specific noise level; when it encounters the variable noise in real cryo-ET data, its features are disrupted. NGM bridges this gap by **injecting target-like noise into the source data**, forcing the student network to be robust to the actual noise it will encounter.

### Why Gaussian Noise?

Deep learning models are more sensitive to noise that conforms to a probability distribution (Lehtinen et al., 2018 – *Noise2Noise*). Gaussian noise provides a tractable, well-understood distribution that can be parameterised by a single scalar σ².

### Pipeline

**Step 1 — 3D Discrete Fourier Transform (DFT):**

For a target sample x^t_n ∈ ℝ^{D×H×W}:

```
x̂_n(u, v, ζ) = ξ[x^t_n]
```

where ξ denotes the 3D DFT. The spatial frequencies (u, v, ζ) parametrise the Fourier domain. The DFT decomposes the volume into sinusoidal components at different frequencies.

**Step 2 — High-Pass Filtering:**

```
x̂'_n(u, v, ζ) = H_high(u, v, ζ) · x̂_n(u, v, ζ)
```

where H_high is a binary mask that **keeps** only the outer ρ fraction of frequencies:

```
H_high(u, v, ζ) = 1   if r(u,v,ζ) > threshold
                  0   otherwise

r(u,v,ζ) = √[(u/D)² + (v/H)² + (ζ/W)²]    (normalised radial frequency)
threshold = √(1 − ρ) / 2
```

**Why this works:**
- **Low frequencies** (inner sphere): carry the smooth, large-scale structural signal of the macromolecule (shape, density gradients)
- **High frequencies** (outer ring): carry noise — random, rapidly-varying patterns with little geometric structure

The filter rate ρ = 24.4% means only 24.4% of frequency components (the highest frequencies, most noise-containing) are kept.

**Step 3 — Inverse DFT:**

```
x^{t'}_{n} = ξ^{-1}[x̂'_n]
```

This gives a spatial-domain estimate of the noise component in x^t_n.

**Step 4 — Average across N_sampled:**

```
x̄^{t'} = (1/N) · Σ_{n=1}^{N} x^{t'}_n
```

Using multiple samples (N=10) reduces variance in the noise estimate from any single outlier volume.

**Step 5 — Variance Estimation and Gaussian Sampling:**

```
σ²_t = Var(x̄^{t'})
ε ~ N(0, σ²_t · I)
```

We only use the variance σ²_t, not the actual extracted noise, to sample a fresh Gaussian noise tensor. This is because:
1. Gaussian noise is a better distributional prior for DL model sensitivity
2. The extracted noise still contains residual signal components
3. Gaussian noise is easily scalable to any spatial resolution

**Step 6 — Source Augmentation:**

```
x^{s'}_i = x^s_i + ε
```

The noisy source volume is used alongside the clean source in the consistency loss, but the segmentation loss only uses the clean source (which has ground-truth labels).

---

## 5. Denoised Pseudo-Labeling (DPL): BF → IBF

### Why Denoising Before Pseudo-Labeling?

The teacher network generates pseudo-labels for target volumes. However, target volumes are noisy, and this noise causes the teacher's predictions to be distorted — sharp, clean boundaries in the segmentation map get blurred or misplaced. These distorted pseudo-labels then *harm* the student's training on the target domain.

The solution: **denoise the target volume before passing it to the teacher**, so the teacher sees a cleaner input and produces less-distorted pseudo-labels.

### Method 1: Bilateral Filter (BF)

The BF is a classical edge-preserving denoiser. It processes the volume with a sliding 3×3×3 window.

**Formula for updated voxel p:**

```
v'_p = Σ_{q ∈ V}  G_σd(||p−q||) · G_σr(v_q − v_p) · v_q
       ─────────────────────────────────────────────────────
       Σ_{q ∈ V}  G_σd(||p−q||) · G_σr(v_q − v_p)
```

**Components:**

**Domain (spatial) kernel:**
```
G_σd(||p−q||) = exp(−||p−q||² / (2σ²_d))
```
- ||p−q|| = Euclidean distance between voxel positions p and q
- σ_d = 120 (spatial bandwidth)
- This kernel gives more weight to spatially nearby voxels (standard Gaussian smoothing)

**Range (intensity) kernel:**
```
G_σr(v_q − v_p) = exp(−(v_q − v_p)² / (2σ²_r))
```
- v_q, v_p = intensity values at voxels q and p
- σ_r = 1.2 (intensity bandwidth)
- This kernel gives high weight to voxels with *similar intensity* to the centre voxel
- Key property: voxels across an edge have very different intensities → low weight → edge preserved

**Gaussian kernel function:**
```
G_σ(x) = exp(−x² / (2σ²))    [normalisation constant omitted; cancels in ratio]
```

**Why normalise (divide by weight sum)?**
The bilateral filter is not a convolution (weights depend on data values), so it's not energy-preserving by construction. Dividing by the weight sum ensures the filtered value is a proper weighted average of neighbourhood intensities.

### Method 2: NGM-based Denoising

Simply subtract the extracted noise: x̃^t = x^t − x^{t'}

Less effective because DFT filtering also removes some edge information along with noise.

### Method 3: Improved Bilateral Filter (IBF) ← Best performing

**The key insight:** In greyscale cryo-ET volumes, raw intensity values are poor discriminators between edges and noise. Two voxels at an edge may have similar intensities but very different *local gradient structures*. The IBF replaces the intensity difference in the range kernel with a **gradient difference computed via the Laplace/Sobel operator**.

#### 3D Laplace Operator

The discrete 3D Laplacian at voxel (h, w, d) is:

```
Δ(v)_{h,w,d} = v_{h+1,w,d} + v_{h-1,w,d}     (h-axis second difference)
              + v_{h,w+1,d} + v_{h,w-1,d}     (w-axis second difference)
              + v_{h,w,d+1} + v_{h,w,d-1}     (d-axis second difference)
              − 6 · v_{h,w,d}                  (centre voxel subtracted 6×)
```

This measures the *local curvature* of the intensity surface. At flat regions, Δ ≈ 0. At edges, Δ has large magnitude.

**Note:** The paper states "We use Sobel operator as the Laplacian operator." The 3D Sobel operator computes directional first derivatives as a separable convolution, approximating the spatial gradient. In our implementation, we compute per-direction Sobel gradients (h, w, d) and use these as the "gradient" vectors for the IBF range kernel.

#### 3D Gradient of Laplacian

For each voxel q, the gradient-of-Laplacian vector is the finite difference of Δ in each direction:

```
∂Δ/∂v^h_q = Δ(v_{q+1}^h, v_q^w, v_q^d) − Δ(v_{q-1}^h, v_q^w, v_q^d)    [Eq.10]
∂Δ/∂v^w_q = Δ(v_q^h, v_{q+1}^w, v_q^d) − Δ(v_q^h, v_{q-1}^w, v_q^d)    [Eq.11]
∂Δ/∂v^d_q = Δ(v_q^h, v_q^w, v_{q+1}^d) − Δ(v_q^h, v_q^w, v_{q-1}^d)    [Eq.12]
```

This gives a 3D vector  ∂Δ/∂v_q = (∂Δ/∂v^h_q,  ∂Δ/∂v^w_q,  ∂Δ/∂v^d_q)  for each voxel.

The gradient difference between centre p and neighbour q:

```
∂Δ/∂v_p − ∂Δ/∂v_q = (∂Δ/∂v^h_p − ∂Δ/∂v^h_q,
                       ∂Δ/∂v^w_p − ∂Δ/∂v^w_q,     [Eq.14]
                       ∂Δ/∂v^d_p − ∂Δ/∂v^d_q)
```

#### IBF Formula

```
v'_q = Σ_{q ∈ V}  G_σd(||p−q||) · G_σr(∂Δ/∂v_p − ∂Δ/∂v_q) · v_q     [Eq.13]
       ────────────────────────────────────────────────────────────────
       Σ_{q ∈ V}  G_σd(||p−q||) · G_σr(∂Δ/∂v_p − ∂Δ/∂v_q)
```

where the range kernel now uses the L2 norm of the gradient-difference vector:

```
G_σr(∂Δ/∂v_p − ∂Δ/∂v_q) = exp(−||∂Δ/∂v_p − ∂Δ/∂v_q||² / (2σ²_r))
```

#### Why Gradient Differences are Better

| Scenario | BF behaviour | IBF behaviour |
|----------|-------------|---------------|
| Two voxels, same object, similar intensity | High range weight ✓ | High range weight ✓ |
| Two voxels across an edge, very different intensity | Low range weight ✓ | Low range weight ✓ |
| Two voxels, same object, **different brightness** (large global gradient) | **Low range weight ✗** (mistakenly treats as edge) | High range weight ✓ (similar gradients → same object) |
| Two voxels across edge, **similar intensity** (iso-intensity boundary) | **High range weight ✗** (fails to detect edge) | Low range weight ✓ (different gradients → edge detected) |

The gradient-based range kernel is specifically sensitive to the local *structural discontinuities* that define object boundaries in cryo-ET, regardless of absolute brightness.

#### Why This Produces Better Pseudo-Labels

When the IBF is applied to the noisy target volume:
1. Noise is suppressed (Gaussian noise has random gradients → low range weight from their neighbours → smoothed away)
2. True macromolecular edges are preserved (sharp gradient changes are maintained)
3. The teacher sees a cleaner, better-defined boundary structure
4. Teacher pseudo-labels align more accurately with true macromolecular surfaces
5. The student learns better target-domain segmentation from less-distorted supervision

#### Ablation Results (from paper)

| Method      | mIoU (Table 1) | Dice (Table 1) |
|-------------|----------------|----------------|
| NGM only    | 48.5           | 64.4           |
| + BF        | 49.1 (+0.6)    | 64.5 (+0.1)    |
| + IBF       | **50.3 (+2.4)**| **65.9 (+3.3)**|

IBF outperforms BF by a larger margin on Dice (3.3%) than mIoU (2.4%), suggesting it particularly improves binary segmentation boundary quality.

---

## 6. Student–Teacher EMA Framework

### Architecture

Both student and teacher are identical VoxResNet instances.

- **Student**: parameters θ_s, updated by gradient descent on all 4 losses
- **Teacher**: parameters θ_t, updated *only* by Exponential Moving Average (EMA) of student

### EMA Update Rule

After each gradient descent step on the student:

```
θ_t ← α · θ_t + (1 − α) · θ_s
```

where α = 0.999 (EMA decay, following FixMatch / Mean Teacher protocols).

**Why EMA?**  
The teacher provides pseudo-labels for the target domain. If teacher = student, its pseudo-labels are noisy and unstable (the student is still learning). EMA smooths the teacher's parameter trajectory:
- The teacher is a *temporal ensemble* of recent student checkpoints
- Short-term fluctuations (batch noise) are damped out
- The teacher is consistently better-calibrated than the instantaneous student
- This breaks the feedback loop of "bad pseudo-labels → bad training → even worse pseudo-labels"

The teacher is initialised as an exact copy of the student at the start of training, ensuring the EMA starts from a meaningful point.

### Pseudo-Label Generation

The teacher generates pseudo-labels for IBF-denoised target volumes:

```python
with torch.no_grad():
    denoised = IBF(x^t_j)
    teacher_logits = teacher(denoised)
    probs = softmax(teacher_logits)         # (B, 2, D, H, W)
    confidence, ỹ^t = probs.max(dim=1)     # (B, D, H, W) each
    mask = confidence ≥ η                  # η = 0.85 threshold
```

Only voxels where the teacher's confidence exceeds η = 0.85 are used in L_pseudo. This prevents error propagation from uncertain predictions.

---

## 7. Loss Functions: Full Derivation

### L_seg — Supervised Segmentation Loss

Applied to the student's output on **clean source** subtomograms with **ground-truth** binary masks.

```
L_seg = L_CE + L_Dice
```

**Cross-Entropy (CE) loss:**
```
L_CE = −(1/|V|) Σ_{v∈V} Σ_{c∈{0,1}} y_{v,c} · log p_{v,c}
```
where:
- V = set of all voxels
- y_{v,c} ∈ {0,1} = one-hot ground-truth for voxel v, class c
- p_{v,c} = softmax(logit_{v,c}) = predicted probability

**Soft Dice loss:**
```
L_Dice = 1 − (2 · Σ_v p_{v,1} · y_{v,1} + ε) / (Σ_v p_{v,1} + Σ_v y_{v,1} + ε)
```
where:
- p_{v,1} = predicted foreground probability at voxel v
- y_{v,1} = ground-truth foreground label (0 or 1)
- ε = 1 (Laplace smoothing to prevent division by zero)

**Why both CE + Dice?**  
CE alone is sensitive to class imbalance (background >> foreground in most subtomograms). Dice loss explicitly maximises overlap between prediction and ground-truth, making it robust to imbalance. Their combination leverages the strengths of both.

### L_con — Feature Consistency Loss

Enforces noise-robustness by requiring similar internal representations for clean and noisy-augmented source volumes.

```
L_con = λ₁·L_BN(fc, f'c)
      + λ₂·L_BN(fv2, f'v2)
      + λ₃·L_BN(fv4, f'v4)
      + λ₄·L_BN(fv6, f'v6)
```

where f_* = student features from **clean** x^s, and f'_* = student features from **noisy** x^{s'} = x^s + ε.

**L_BN (cosine similarity loss):**
```
L_BN(f, f') = 1 − mean_{v∈spatial} [cos_sim(f_v, f'_v)]

cos_sim(f_v, f'_v) = (f_v · f'_v) / (||f_v|| · ||f'_v||)   ∈ [−1, 1]
```

where f_v is the channel vector at voxel position v.

**Layer weights:** [λ₁=0.2, λ₂=0.2, λ₃=0.3, λ₄=0.3]

The paper justifies this weighting by noting:
- Shallower layers (fc, fv2) capture textural details that change noticeably with noise → lower weight because perfect consistency is unrealistic
- Deeper layers (fv4, fv6) capture structural/edge information that should be noise-invariant → higher weight as the primary training signal for robustness

**Ablation table (from paper):**

| λ₁ → λ₄             | mIoU  | Dice  |
|----------------------|-------|-------|
| [0.1, 0.1, 0.4, 0.4] | 44.4  | 60.2  |
| **[0.2, 0.2, 0.3, 0.3]** | **50.3** | **65.9** |
| [0.3, 0.3, 0.2, 0.2] | 45.3  | 61.0  |
| [0.4, 0.4, 0.1, 0.1] | 42.2  | 57.8  |

Balanced weights with slight emphasis on deep layers is optimal.

### L_dis — Domain Discriminator Loss

The discriminator D takes the deep feature map fv6 and predicts domain membership.

```
L_dis = L_BCE(D(fv6^s), 0) + L_BCE(D(fv6^t), 1)
```

where:
- 0 = source domain label
- 1 = target domain label
- L_BCE(logit, y) = −[y·log σ(logit) + (1−y)·log(1−σ(logit))]

**Gradient Reversal Layer (GRL):**

The GRL is inserted between the encoder's fv6 output and the discriminator. In the forward pass it is an identity; in the backward pass, gradients are negated:

```
Forward:  output = input
Backward: ∂L/∂input = −λ · ∂L/∂output
```

Effect:
- Discriminator minimises L_dis → correctly classifies source vs. target
- Encoder *maximises* L_dis → produces features indistinguishable across domains
- These two objectives are simultaneously satisfied via the sign flip

This is the DANN (Domain Adversarial Neural Network) mechanism from Ganin et al. (2016).

### L_pseudo — Pseudo-Label Loss

Applied to the student's output on **unlabelled target** volumes, supervised by teacher-generated pseudo-labels.

```
L_pseudo = L_CE(student(x^t_j), ỹ^t_j)   [only on mask voxels where conf ≥ η]
```

**Confidence thresholding** (η = 0.85):
- Voxels where teacher confidence < 0.85 are masked out (set to ignore_index = -1)
- Only high-confidence pseudo-labels contribute to the gradient
- Prevents error accumulation from uncertain predictions in early training

### Total Loss

```
L = L_seg + α·L_con + β·L_dis + γ·L_pseudo
```

Default: α = β = γ = 1.0 (equal weighting, as implied by paper Eq. 1).

---

## 8. Domain Discriminator and Adversarial Training

### Architecture

```
fv6: (B, 64, D, H, W)
     ↓ GRL(λ=1.0)
     ↓ Conv3d(64→64, 3×3×3) + BN + LeakyReLU(0.2)
     ↓ Conv3d(64→32, 3×3×3) + BN + LeakyReLU(0.2)
     ↓ Conv3d(32→1, 1×1×1)
     ↓ GlobalAvgPool3D
domain_logit: (B, 1)   ← scalar per sample
```

### Training Dynamics

**Iteration t:**
1. Compute fv6^s from student on source batch
2. Compute fv6^t from student on target batch
3. Forward both through discriminator → scalar logits
4. Compute L_dis (BCE)
5. Backward propagates through GRL → encoder sees REVERSED gradient

**Net effect over training:**
- Discriminator gets better at distinguishing source/target → L_dis decreases for D
- Encoder gets better at hiding domain information → L_dis *cannot* decrease for encoder
- Equilibrium: encoder produces domain-invariant features

---

## 9. Training Flow: Step-by-Step

Each training batch consists of:
- One batch of source: (x^s, y^s) — (B, 1, 32, 32, 32) + (B, 32, 32, 32)
- One batch of target: x^t — (B, 1, 32, 32, 32)
- A subset of N=10 target volumes: target_subset — (10, 1, 32, 32, 32)

**Detailed forward pass:**

```
Step 1:  NGM(source, target_subset)
         → extract high-freq noise from target_subset via DFT
         → estimate σ²_t = Var(avg noise)
         → ε ~ N(0, σ²_t)
         → x^{s'} = x^s + ε

Step 2:  Student(x^s)  → {logits_s, fc_s, fv2_s, fv4_s, fv6_s}
         Student(x^{s'}) → {logits_n, fc_n, fv2_n, fv4_n, fv6_n}
         Student(x^t)   → {logits_t,          fv6_t}

Step 3:  Discriminator(fv6_s)  → domain_logit_s   (with GRL)
         Discriminator(fv6_t)  → domain_logit_t   (with GRL)

Step 4:  [no_grad]
         x̃^t = IBF(x^t)          (denoised target)
         Teacher(x̃^t) → teacher_logits
         ỹ^t, mask = pseudo_labels(teacher_logits, η=0.85)

Step 5:  L_seg    = CE(logits_s, y^s) + Dice(logits_s, y^s)
         L_con    = Σ_k λ_k · (1 − cos_sim(fc_s_k, fc_n_k))
         L_dis    = BCE(domain_logit_s, 0) + BCE(domain_logit_t, 1)
         L_pseudo = CE(logits_t, ỹ^t)[mask]
         L_total  = L_seg + L_con + L_dis + L_pseudo

Step 6:  L_total.backward()
         optimizer.step()   (Adam, lr=1e-3)

Step 7:  EMA update:  θ_t ← 0.999·θ_t + 0.001·θ_s
```

**Learning rate schedule:**
```
lr at epoch e = 1e-3 × (0.1)^⌊e/100⌋
```
- Epoch 1–100:   lr = 1.0e-3
- Epoch 101–200: lr = 1.0e-4
- Epoch 201–300: lr = 1.0e-5

---

## 10. Hyperparameter Guide

| Parameter | Default | Meaning | Ablation result |
|-----------|---------|---------|-----------------|
| N_sampled | 10 | # target volumes for NGM noise estimation | 5→41.2, **10→50.3**, 15→43.6, 20→42.6 |
| ρ (filter_rate) | 0.244 (24.4%) | Fraction of high-freq kept in NGM HPF | 8.4%→41.7, 17.8%→43.5, **24.4%→50.3**, 42.2%→41.0 |
| λ weights | [0.2,0.2,0.3,0.3] | L_con layer weights | See Section 7 table |
| σ_d | 120 | IBF spatial bandwidth | 100→49.3, **120→50.3**, 140→49.1, 160→47.0 |
| σ_r | 1.2 | IBF range (gradient) bandwidth | 0.8→47.5, 1.0→48.0, **1.2→50.3**, 1.4→49.5 |
| η | 0.85 | Pseudo-label confidence threshold | Paper default, not ablated |
| α_EMA | 0.999 | Teacher EMA decay | Standard FixMatch default |

**Key observations from ablation:**
- N_sampled=10 is a Goldilocks number: too few (5) → poor noise estimate; too many (15+) → averaging dilutes signal
- ρ=24.4% keeps just enough frequency information to capture noise without structural content
- σ_d=120 is very large (spatial window up to 3 voxels), emphasising spatial proximity strongly
- σ_r=1.2 is moderate: too small ignores too many neighbours, too large loses edge sensitivity

---

## 11. Repository Structure

```
vox_uda/
├── models/
│   ├── voxresnet.py        VoxResNet backbone + VoxResBlock + EMA helper
│   └── discriminator.py    DomainDiscriminator + GradReversalLayer
├── modules/
│   ├── ngm.py              Noise Generation Module (DFT + HPF + Gaussian)
│   └── ibf.py              BilateralFilter3D, ImprovedBilateralFilter3D
├── losses/
│   └── losses.py           SegmentationLoss, ConsistencyLoss, DiscriminatorLoss,
│                           PseudoLabelLoss, VoxUDALoss
├── datasets/
│   └── cryoet_dataset.py   SourceDataset, TargetDataset, TargetNoiseSubset (will be implemented)
├── utils/
│   └── metrics.py          SegmentationMetrics (mIoU, Dice)
├── configs/
│   └── config.py           VoxUDAConfig dataclass
├── vox_uda.py              Full VoxUDA model integration
├── train.py                Training script (will be implemented)
├── inference.py            Inference + evaluation script (will be implemented)
└── DOCUMENTATION.md        This file
```

---

## 12. Usage Guide


### Data Preparation (will be implemented)

Organise your data as (Hypothetical structure, currently):

```
data/
├── source/
│   ├── volumes/  # *.npy files, shape (32,32,32), float32
│   └── masks/    # *.npy files, shape (32,32,32), binary {0,1}
└── target/
    ├── volumes/  # *.npy files, shape (32,32,32), float32 (NO labels)
    ├── eval/     # (optional) labelled subset for monitoring
    └── masks/    # (optional) corresponding masks for eval
```

For large subtomograms (e.g., Mycoplasma 192³), pre-cut them into 32³ patches:

```python
import numpy as np

vol = np.load("large_subtomogram.npy")   # (192, 192, 192)
patches = []
for i in range(0, 192, 32):
    for j in range(0, 192, 32):
        for k in range(0, 192, 32):
            patch = vol[i:i+32, j:j+32, k:k+32]
            patches.append(patch)
# Save patches as individual .npy files
```

### Training (Hypothetical)

```bash
# Default configuration (IBF denoiser, paper hyperparameters)
python train.py \
    --source_volume_dir data/source/volumes \
    --source_mask_dir   data/source/masks \
    --target_volume_dir data/target/volumes \
    --denoiser ibf \
    --epochs 300

# With BF denoiser
python train.py --denoiser bf

# With NGM-only denoising
python train.py --denoiser ngm
```

