"""
Bilateral Filter (BF) and Improved Bilateral Filter (IBF) for 3D Cryo-ET.

Background
----------
The bilateral filter is a non-linear, edge-preserving smoothing filter.
Unlike Gaussian smoothing (which blurs edges), the bilateral filter weights
each neighbour by BOTH spatial proximity AND signal similarity, so it
smooths flat regions while preserving sharp boundaries.

Standard Bilateral Filter (BF)
-------------------------------
For a 3D greyscale volume, the filtered value at the central voxel p is:

    v'_p = Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(v_q - v_p) · v_q
           ────────────────────────────────────────────────────
           Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(v_q - v_p)

where:
    V      = 3×3×3 neighbourhood of p (27 voxels)
    G_σ(x) = exp(-x² / (2σ²)) / (2π σ²)   – Gaussian kernel
    ||p-q||= Euclidean distance between voxel positions p and q
    σ_d    = domain hyperparameter (spatial width, 120 in paper)
    σ_r    = range hyperparameter  (intensity width, 1.2 in paper)

The DOMAIN filter  G_σd(||p-q||)   weights by spatial proximity.
The RANGE filter   G_σr(v_q - v_p) weights by intensity similarity
                                    → preserves edges.

Limitation of BF in Greyscale
------------------------------
The range kernel G_σr(v_q - v_p) compares raw voxel intensities.  In an
RGB image this works well, but in a greyscale cryo-ET volume there are large
global brightness variations unrelated to edges.  The range kernel can then
mistake brightness differences for edge information and vice-versa.

Improved Bilateral Filter (IBF)
--------------------------------
The IBF replaces the intensity difference in the range kernel with the
GRADIENT difference, computed via the 3D Laplace/Sobel operator.

Intuition:  Voxels that belong to the SAME object (interior) share similar
local gradients; voxels across an EDGE have very different gradients.  So
using gradient differences makes the range kernel more sensitive to actual
structural boundaries and less sensitive to global brightness offsets.

IBF formula:

    v'_q = Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(∂Δ/∂v_p − ∂Δ/∂v_q) · v_q
           ─────────────────────────────────────────────────────────────
           Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(∂Δ/∂v_p − ∂Δ/∂v_q)

where ∂Δ/∂v_q is the 3D gradient-of-Laplacian vector at voxel q:

    ∂Δ/∂v^h_q = Δ(v_{q+1}^h, v_q^w, v_q^d) − Δ(v_{q-1}^h, v_q^w, v_q^d)
    ∂Δ/∂v^w_q = Δ(v_q^h, v_{q+1}^w, v_q^d) − Δ(v_q^h, v_{q-1}^w, v_q^d)
    ∂Δ/∂v^d_q = Δ(v_q^h, v_q^w, v_{q+1}^d) − Δ(v_q^h, v_q^w, v_{q-1}^d)

Here Δ denotes the Laplace operator (implemented with 3D Sobel in the paper).

The gradient-difference vector in the range kernel is:
    ∂Δ/∂v_p − ∂Δ/∂v_q = (Δ∂h_p − Δ∂h_q,  Δ∂w_p − Δ∂w_q,  Δ∂d_p − Δ∂d_q)

and its norm is used in G_σr.

Implementation Note
-------------------
Computing the bilateral filter voxel-by-voxel in Python loops is very slow.
We use an unfolding trick: extract all 27 neighbours of every voxel as a
tensor, then compute weights and weighted averages in parallel over the
spatial dimensions.

Paper hyperparameters:
    σ_d = 120,  σ_r = 1.2,  window = 3×3×3
    Sobel used as Laplacian operator approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D Sobel / Laplacian gradient operator
# ---------------------------------------------------------------------------

def compute_3d_sobel_gradients(volume: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3D Sobel gradient of a greyscale volume.

    The paper states: "We use Sobel operator as the Laplacian operator."
    This function computes the first-order Sobel derivative in h, w, d
    directions, which approximates the gradient of the Laplacian operator.

    Args:
        volume: (B, 1, D, H, W) float tensor

    Returns:
        grad: (B, 3, D, H, W)
              channel 0 = ∂/∂h (depth gradient)
              channel 1 = ∂/∂w (height gradient)
              channel 2 = ∂/∂d (width gradient)
    """
    # 3D Sobel kernels (separable form: outer product of [1,2,1], [1,0,-1])
    # We construct one kernel per axis direction.

    # Smoothing kernel along an axis: [1, 2, 1] / 4
    smooth = torch.tensor([1., 2., 1.], device=volume.device) / 4.0
    # Derivative kernel: [1, 0, -1]
    deriv  = torch.tensor([1., 0., -1.], device=volume.device)

    def outer3(a, b, c):
        """Outer product of three 1D tensors → 3×3×3 kernel."""
        return torch.einsum("i,j,k->ijk", a, b, c)

    # Sobel for h (depth) direction: derivative along h, smooth along w and d
    Kh = outer3(deriv, smooth, smooth).view(1, 1, 3, 3, 3).to(volume.device)
    # Sobel for w (height) direction
    Kw = outer3(smooth, deriv, smooth).view(1, 1, 3, 3, 3).to(volume.device)
    # Sobel for d (width) direction
    Kd = outer3(smooth, smooth, deriv).view(1, 1, 3, 3, 3).to(volume.device)

    grad_h = F.conv3d(volume, Kh, padding=1)   # (B, 1, D, H, W)
    grad_w = F.conv3d(volume, Kw, padding=1)
    grad_d = F.conv3d(volume, Kd, padding=1)

    return torch.cat([grad_h, grad_w, grad_d], dim=1)   # (B, 3, D, H, W)


def compute_3d_laplacian(volume: torch.Tensor) -> torch.Tensor:
    """
    3D discrete Laplacian Δ(v) = sum of second differences along all axes.

        Δ(v)_ijk = v_{i+1,j,k} + v_{i-1,j,k}
                 + v_{i,j+1,k} + v_{i,j-1,k}
                 + v_{i,j,k+1} + v_{i,j,k-1}  − 6·v_{i,j,k}

    Args:
        volume: (B, 1, D, H, W)
    Returns:
        lap: (B, 1, D, H, W)
    """
    kernel = torch.zeros(1, 1, 3, 3, 3, device=volume.device)
    kernel[0, 0, 1, 1, 1] = -6.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    return F.conv3d(volume, kernel, padding=1)


# ---------------------------------------------------------------------------
# Gaussian kernel helper
# ---------------------------------------------------------------------------

def gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Element-wise Gaussian kernel:  G_σ(x) = exp(-x² / (2σ²))

    Note: we omit the normalisation constant 1/(2πσ²) because the
    bilateral filter normalises by the weight sum anyway.
    """
    return torch.exp(-x ** 2 / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Bilateral Filter (BF) – standard, intensity-based range kernel
# ---------------------------------------------------------------------------

class BilateralFilter3D(nn.Module):
    """
    Standard 3D Bilateral Filter with intensity-based range kernel.

    BF(v_q) =  Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(v_q − v_p) · v_q
               ─────────────────────────────────────────────────────
               Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(v_q − v_p)

    Args:
        sigma_d : domain (spatial) bandwidth  (default 120).
        sigma_r : range  (intensity) bandwidth (default 1.2).
        window  : local window side length (default 3 → 3×3×3 patch).
    """

    def __init__(self, sigma_d: float = 120.0, sigma_r: float = 1.2, window: int = 3):
        super().__init__()
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.window  = window
        self.pad     = window // 2

        # Pre-compute spatial distance weights for all offsets in the window.
        # Shape: (window^3,) – one weight per neighbour position.
        offsets = []
        for dh in range(-self.pad, self.pad + 1):
            for dw in range(-self.pad, self.pad + 1):
                for dd in range(-self.pad, self.pad + 1):
                    offsets.append([dh, dw, dd])
        offsets = torch.tensor(offsets, dtype=torch.float32)   # (K, 3)
        dist_sq = (offsets ** 2).sum(dim=1)                    # (K,)
        # G_σd(||p-q||) for each offset
        spatial_weights = torch.exp(-dist_sq / (2.0 * sigma_d ** 2))  # (K,)
        # Register as buffer (moves to GPU automatically, not a parameter)
        self.register_buffer("spatial_weights", spatial_weights)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: (B, 1, D, H, W) input greyscale volume.
        Returns:
            denoised: (B, 1, D, H, W) bilaterally filtered volume.
        """
        B, C, D, H, W = volume.shape
        assert C == 1, "BilateralFilter3D expects single-channel (greyscale) input."

        pad = self.pad
        # Pad to handle borders (replicate padding avoids edge artefacts)
        vol_pad = F.pad(volume, [pad]*6, mode="replicate")   # (B,1,D+2p,H+2p,W+2p)

        # Unfold: extract all window^3 neighbour values for every voxel
        # F.unfold doesn't support 3D, so we do it manually with slicing
        K = self.window ** 3
        neighbors = torch.zeros(B, K, D, H, W, device=volume.device)

        k = 0
        for dh in range(self.window):
            for dw in range(self.window):
                for dd in range(self.window):
                    neighbors[:, k] = vol_pad[:, 0,
                                              dh:dh+D,
                                              dw:dw+H,
                                              dd:dd+W]
                    k += 1
        # neighbors: (B, K, D, H, W)

        # Central voxel values for every spatial position
        center = volume[:, 0].unsqueeze(1)              # (B, 1, D, H, W)

        # Range weights: G_σr(v_q - v_p)
        intensity_diff    = neighbors - center           # (B, K, D, H, W)
        range_weights     = gaussian(intensity_diff, self.sigma_r)

        # Combine domain and range weights
        # spatial_weights: (K,) → broadcast to (B, K, D, H, W)
        sw = self.spatial_weights.view(1, K, 1, 1, 1)
        weights = sw * range_weights                     # (B, K, D, H, W)

        # Weighted sum
        numerator   = (weights * neighbors).sum(dim=1, keepdim=True)
        denominator = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return numerator / denominator                   # (B, 1, D, H, W)


# ---------------------------------------------------------------------------
# Improved Bilateral Filter (IBF) – gradient-based range kernel
# ---------------------------------------------------------------------------

class ImprovedBilateralFilter3D(nn.Module):
    """
    Improved 3D Bilateral Filter with Laplacian-gradient range kernel.

    The key innovation: instead of comparing raw intensity values in the
    range kernel, we compare the 3D gradient-of-Laplacian vectors.

    IBF(v_q) = Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(∂Δ/∂v_p − ∂Δ/∂v_q) · v_q
               ─────────────────────────────────────────────────────────────
               Σ_{q ∈ V}  G_σd(||p-q||) · G_σr(∂Δ/∂v_p − ∂Δ/∂v_q)

    where ∂Δ/∂v_q = 3D Sobel gradient vector at q  (h, w, d components).

    The gradient difference used in G_σr is the L2 norm of the vector:
        ||∂Δ/∂v_p − ∂Δ/∂v_q||²  (summed over h, w, d channels)

    Args:
        sigma_d : domain (spatial) bandwidth  (default 120).
        sigma_r : range (gradient) bandwidth  (default 1.2).
        window  : local window side length (default 3).
    """

    def __init__(self, sigma_d: float = 120.0, sigma_r: float = 1.2, window: int = 3):
        super().__init__()
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.window  = window
        self.pad     = window // 2

        # Pre-compute spatial distance weights
        offsets = []
        for dh in range(-self.pad, self.pad + 1):
            for dw in range(-self.pad, self.pad + 1):
                for dd in range(-self.pad, self.pad + 1):
                    offsets.append([dh, dw, dd])
        offsets = torch.tensor(offsets, dtype=torch.float32)
        dist_sq = (offsets ** 2).sum(dim=1)
        spatial_weights = torch.exp(-dist_sq / (2.0 * sigma_d ** 2))
        self.register_buffer("spatial_weights", spatial_weights)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: (B, 1, D, H, W) input greyscale volume.
        Returns:
            denoised: (B, 1, D, H, W) IBF-filtered volume.
        """
        B, C, D, H, W = volume.shape
        assert C == 1, "ImprovedBilateralFilter3D expects single-channel input."

        pad = self.pad

        # ---- Step 1: Compute 3D Sobel gradient map ----
        # grad: (B, 3, D, H, W)  –  3 channels for h, w, d gradient components
        grad = compute_3d_sobel_gradients(volume)     # uses Sobel as Laplacian approx

        # ---- Step 2: Pad volume and gradient for neighbour extraction ----
        vol_pad  = F.pad(volume, [pad]*6, mode="replicate")  # (B,1,D+2p,H+2p,W+2p)
        grad_pad = F.pad(grad,   [pad]*6, mode="replicate")  # (B,3,D+2p,H+2p,W+2p)

        K = self.window ** 3

        # Neighbour intensity values
        neighbor_vals  = torch.zeros(B, K,    D, H, W, device=volume.device)
        # Neighbour gradient vectors: 3 channels per neighbour → (B, K*3, D, H, W)
        neighbor_grads = torch.zeros(B, K, 3, D, H, W, device=volume.device)

        k = 0
        for dh in range(self.window):
            for dw in range(self.window):
                for dd in range(self.window):
                    neighbor_vals[:, k]     = vol_pad[:, 0,
                                                       dh:dh+D, dw:dw+H, dd:dd+W]
                    neighbor_grads[:, k, :] = grad_pad[:, :,
                                                        dh:dh+D, dw:dw+H, dd:dd+W]
                    k += 1
        # neighbor_vals:  (B, K, D, H, W)
        # neighbor_grads: (B, K, 3, D, H, W)

        # ---- Step 3: Gradient of central voxel ----
        center_grad = grad.unsqueeze(1)    # (B, 1, 3, D, H, W)

        # ---- Step 4: Gradient difference vector ----
        # (B, K, 3, D, H, W) – difference between central and neighbour gradient
        grad_diff = center_grad - neighbor_grads

        # L2 norm of gradient-difference vector over the 3 spatial components
        grad_diff_norm = grad_diff.pow(2).sum(dim=2).sqrt()   # (B, K, D, H, W)

        # ---- Step 5: Range weights using gradient norm (not intensity) ----
        range_weights = gaussian(grad_diff_norm, self.sigma_r)  # (B, K, D, H, W)

        # ---- Step 6: Combine domain × range weights ----
        sw      = self.spatial_weights.view(1, K, 1, 1, 1)
        weights = sw * range_weights                             # (B, K, D, H, W)

        # ---- Step 7: Weighted average (over intensity values, not gradients) ----
        numerator   = (weights * neighbor_vals).sum(dim=1, keepdim=True)
        denominator = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return numerator / denominator                           # (B, 1, D, H, W)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_denoiser(
    method: str,
    sigma_d: float = 120.0,
    sigma_r: float = 1.2,
    window: int = 3,
) -> nn.Module:
    """
    Factory that returns the requested denoiser module.

    Args:
        method  : one of "ibf" (improved bilateral filter),
                         "bf"  (standard bilateral filter).
        sigma_d : domain bandwidth.
        sigma_r : range bandwidth.
        window  : sliding window side length.

    Returns:
        nn.Module with a .forward(volume) -> denoised method.
    """
    if method == "ibf":
        return ImprovedBilateralFilter3D(sigma_d=sigma_d, sigma_r=sigma_r, window=window)
    elif method == "bf":
        return BilateralFilter3D(sigma_d=sigma_d, sigma_r=sigma_r, window=window)
    else:
        raise ValueError(f"Unknown denoiser method '{method}'. Choose 'ibf' or 'bf'.")