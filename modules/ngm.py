"""
Noise Generation Module (NGM) for Vox-UDA.

The figure shows the following pipeline for N_sampled target volumes:

    ┌─────────────────────────────────────────────────────────┐
    │  Target subset (N volumes)                              │
    │                                                         │
    │  x^t_1 ──► DFT ──► HPF ──► iDFT ──► noise_1 ──► σ₁   │
    │  x^t_2 ──► DFT ──► HPF ──► iDFT ──► noise_2 ──► σ₂   │
    │  ...                                          ...       │
    │  x^t_N ──► DFT ──► HPF ──► iDFT ──► noise_N ──► σₙ   │
    │                                                         │
    │  ε₁ ~ N(0, σ₁²·I)   [individual Gaussian samples]      │
    │  ε₂ ~ N(0, σ₂²·I)                                      │
    │  ...                                                    │
    │  εₙ ~ N(0, σₙ²·I)                                      │
    │                                                         │
    │  ε_avg = (ε₁ + ε₂ + ... + εₙ) × (1/N)                 │
    │  x^{s'} = x^s + ε_avg         ──► 32×32×32 noisy cube  │
    └─────────────────────────────────────────────────────────┘


Step-by-step:
  For each n = 1 … N_sampled:
    1. DFT:          x̂_n(u,v,ζ) = ξ[x^t_n]
    2. High-pass:    x̂'_n       = H_high · x̂_n   (keep outer ρ fraction)
    3. iDFT:         noise_n     = ξ⁻¹[x̂'_n]
    4. Variance:     σₙ²         = Var(noise_n)
    5. Gaussian:     εₙ ~ N(0, σₙ²·I)
  End for
  6. Average:  ε = (1/N) Σ εₙ
  7. Augment:  x^{s'} = x^s + ε

"""

import torch
import torch.nn as nn


class NoiseGenerationModule(nn.Module):
    """
    NGM: extracts per-sample target noise statistics via DFT + high-pass
    filtering and injects averaged Gaussian noise into source subtomograms.

    Args:
        filter_rate : fraction of high-frequency components to KEEP (ρ = 24.4%).
                      The remaining (1-ρ) low-frequency region is zeroed out.
        n_sampled   : number of target samples used per estimate (N_sampled = 10).
    """

    def __init__(self, filter_rate: float = 0.244, n_sampled: int = 10):
        super().__init__()
        self.filter_rate = filter_rate
        self.n_sampled   = n_sampled

    # ------------------------------------------------------------------
    # High-pass mask builder (shared across all samples in one batch)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_high_pass_mask(
        shape: tuple,
        filter_rate: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        3D ellipsoidal high-pass mask in the Fourier-shifted domain.

        Coordinates are normalised to [-0.5, 0.5] so that the DC component
        is at the centre.  The mask keeps all (u,v,ζ) whose normalised
        distance from DC exceeds a threshold:

            r(u,v,ζ) = √[(u/D)² + (v/H)² + (ζ/W)²]

        Threshold is chosen so that the inner sphere encompasses fraction
        (1 - filter_rate) of the total spectral volume:
            threshold = √(1 - filter_rate) / 2

        At filter_rate = 0.244:  threshold ≈ 0.435
        → inner (low-freq) sphere  = 75.6% of spectrum  → zeroed
        → outer (high-freq) shell  = 24.4% of spectrum  → kept

        Returns:
            mask : (1, 1, D, H, W) float tensor — 1 = keep, 0 = zero
        """
        D, H, W = shape
        d_c = torch.linspace(-0.5, 0.5, D, device=device)
        h_c = torch.linspace(-0.5, 0.5, H, device=device)
        w_c = torch.linspace(-0.5, 0.5, W, device=device)
        zz, yy, xx = torch.meshgrid(d_c, h_c, w_c, indexing="ij")
        r          = torch.sqrt(zz**2 + yy**2 + xx**2)
        threshold  = (1.0 - filter_rate) ** 0.5 / 2.0
        mask       = (r > threshold).float()
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, D, H, W)

    # ------------------------------------------------------------------
    # Per-sample noise extraction
    # ------------------------------------------------------------------

    def _extract_noise_single(
        self,
        vol: torch.Tensor,   # (1, D, H, W) single target volume
        hpf_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the high-frequency noise component from one target volume.

        Steps:
          1. 3D DFT:  x̂ = fftn(vol)
          2. Shift DC to centre: fftshift
          3. Multiply by high-pass mask H_high
          4. Shift back: ifftshift
          5. iDFT:  noise = real(ifftn(masked_freq))

        Returns:
            noise : (1, D, H, W) real-valued noise estimate
        """
        freq       = torch.fft.fftn(vol,           dim=(-3, -2, -1))
        freq_shift = torch.fft.fftshift(freq,      dim=(-3, -2, -1))
        freq_hp    = freq_shift * hpf_mask
        freq_back  = torch.fft.ifftshift(freq_hp,  dim=(-3, -2, -1))
        noise      = torch.fft.ifftn(freq_back,    dim=(-3, -2, -1)).real
        return noise   # (1, D, H, W)

    # ------------------------------------------------------------------
    # Main forward: matches Figure 2 exactly
    # ------------------------------------------------------------------

    def forward(
        self,
        source:        torch.Tensor,   # (B, 1, D, H, W) clean source
        target_subset: torch.Tensor,   # (N, 1, D, H, W) target noise subset
    ) -> dict:
        """
        Full NGM pipeline (Figure 2):
          For each target sample n:
            noise_n  = HPF(DFT(x^t_n))  [high-freq noise estimate]
            σₙ²      = Var(noise_n)
            εₙ ~ N(0, σₙ²·I)            [per-sample Gaussian noise]
          ε_avg = mean(ε₁, ..., εₙ)      [averaged noise volume]
          x^{s'} = x^s + ε_avg

        Returns:
            dict:
              noisy_source : (B, 1, D, H, W)  x^{s'} = x^s + ε_avg
              epsilon      : (B, 1, D, H, W)  the averaged noise ε_avg
              sigma_sq_list: list of N scalar σₙ² values (for diagnostics)
        """
        N, C, D, H, W = target_subset.shape
        device = source.device

        hpf_mask = self._build_high_pass_mask((D, H, W), self.filter_rate, device)

        # --- Per-sample: extract noise → variance → Gaussian sample ---
        epsilon_list = []
        sigma_sq_list = []

        for n in range(N):
            vol    = target_subset[n].to(device)          # (1, D, H, W)
            noise_n = self._extract_noise_single(vol, hpf_mask)  # (1, D, H, W)

            # σₙ² = Var(noise_n)  — scalar
            sigma_sq_n = noise_n.var()
            sigma_sq_list.append(sigma_sq_n.item())

            # εₙ ~ N(0, σₙ²·I)  — same spatial shape as source batch
            eps_n = torch.randn_like(source) * sigma_sq_n.sqrt()
            epsilon_list.append(eps_n)

        # --- Average Gaussian noise samples (×1/N as in Figure 2) ---
        epsilon_avg  = torch.stack(epsilon_list, dim=0).mean(dim=0)   # (B,1,D,H,W)
        noisy_source = source + epsilon_avg

        return {
            "noisy_source":  noisy_source,
            "epsilon":       epsilon_avg,
            "sigma_sq_list": sigma_sq_list,  # one per target sample
        }

    # ------------------------------------------------------------------
    # NGM-based denoiser (alternative to IBF, used when denoiser='ngm')
    # ------------------------------------------------------------------

    def denoise(self, target: torch.Tensor) -> torch.Tensor:
        """
        Estimate and subtract the high-frequency noise from target volumes.

        x̃^t = x^t − noise(x^t)

        Less effective than IBF because DFT-based noise removal also
        discards some low-frequency edge information.

        Args:
            target : (B, 1, D, H, W)
        Returns:
            denoised : (B, 1, D, H, W)
        """
        B, C, D, H, W = target.shape
        device  = target.device
        hpf     = self._build_high_pass_mask((D, H, W), self.filter_rate, device)

        denoised_list = []
        for b in range(B):
            vol     = target[b]                                     # (1, D, H, W)
            noise_b = self._extract_noise_single(vol, hpf)         # (1, D, H, W)
            denoised_list.append(vol - noise_b)

        return torch.stack(denoised_list, dim=0)                    # (B, 1, D, H, W)