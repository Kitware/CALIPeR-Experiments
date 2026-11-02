# Adapted from DAATSim; CC BY-NC 4.0. See DAATSIM_NOTICE.txt.
"""DAATSim phase-screen warping and spatially varying Zernike blur kernels.

The Table 3 simulation adapter uses uniform depth with explicit optical,
turbulence and temporal parameters.
"""
import math

import numpy as np
import scipy.special as sp
import torch
from PIL import Image
from scipy.special import gamma as scipy_gamma
from torch.nn import functional as F

from . import daatsim_config as _default_config


# ---------------------------------------------------------------------------
# Phase-screen generation, Zernike statistics and PSF construction
# ---------------------------------------------------------------------------

_psfGen_cache: dict = {}


def genZernikeCoeff_independent(num_zern, Dr0, batch_size=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.randn(batch_size, num_zern, device=device) * (Dr0 ** (5.0 / 6.0))


def psfGen(N_psf, coeff, wavelength=0.525e-6, D_aper=0.2, z_i=1.2):
    global _psfGen_cache
    device_val = coeff.device
    coeff = coeff.float()
    cache_key = (N_psf, str(device_val))

    if cache_key not in _psfGen_cache:
        x = torch.linspace(-1, 1, N_psf, device=device_val)
        y = torch.linspace(-1, 1, N_psf, device=device_val)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        rho = torch.sqrt(X**2 + Y**2)
        theta = torch.atan2(Y, X)
        mask = rho <= 1
        _psfGen_cache[cache_key] = (rho, theta, mask)
    else:
        rho, theta, mask = _psfGen_cache[cache_key]

    phase = coeff[:, 0:1].reshape(-1, 1, 1) * (2 * rho**2 - 1)
    for i in range(1, coeff.shape[1]):
        phase += coeff[:, i:i+1].reshape(-1, 1, 1) * rho**(i+1) * torch.cos((i+1) * theta)

    wave = torch.zeros(coeff.shape[0], N_psf, N_psf, dtype=torch.complex64, device=device_val)
    mask_expanded = mask.unsqueeze(0).expand(coeff.shape[0], N_psf, N_psf)
    wave = torch.where(mask_expanded, torch.exp(1j * 2 * torch.pi * phase), wave)

    h = torch.fft.fftshift(torch.fft.ifft2(wave, dim=(-2, -1)), dim=(-2, -1))
    psf_raw = torch.abs(h)**2

    grid_y, grid_x = torch.meshgrid(
        torch.arange(N_psf, device=device_val, dtype=psf_raw.dtype),
        torch.arange(N_psf, device=device_val, dtype=psf_raw.dtype),
        indexing='ij')
    grid_y = grid_y.unsqueeze(0).expand_as(psf_raw)
    grid_x = grid_x.unsqueeze(0).expand_as(psf_raw)
    psf_sum_raw = psf_raw.sum(dim=(-2, -1), keepdim=True)
    safe_sum = psf_sum_raw + 1e-12
    centroid_y = (psf_raw * grid_y).sum(dim=(-2, -1), keepdim=True) / safe_sum
    centroid_x = (psf_raw * grid_x).sum(dim=(-2, -1), keepdim=True) / safe_sum
    center = N_psf // 2
    shift_y = centroid_y.squeeze(-1).squeeze(-1) - center
    shift_x = centroid_x.squeeze(-1).squeeze(-1) - center
    centroid_shifts = torch.stack([shift_y, shift_x], dim=-1)
    return psf_raw.float(), centroid_shifts


def custom_resize(img, output_shape):
    is_batched = img.dim() == 4
    orig_dtype = img.dtype
    if not is_batched:
        img = img.float().unsqueeze(0).unsqueeze(0)
    else:
        img = img.float()
    img = F.interpolate(img, size=output_shape, mode='bilinear', align_corners=False)
    if not is_batched:
        img = img.squeeze()
    return img.to(orig_dtype)


def generate_phase_screen(size, r0, L0, l0, screen_size_val, device, scale_factor=4e7):
    delta_x = screen_size_val / size
    fx = torch.fft.fftfreq(size, d=delta_x).to(device)
    fy = torch.fft.fftfreq(size, d=delta_x).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing="ij")
    K_freq = torch.sqrt((2 * torch.pi * FX)**2 + (2 * torch.pi * FY)**2)
    k0_vk = 2 * torch.pi / L0 if L0 != float('inf') else 0.0
    km_vk = 5.92 / l0 if l0 != 0.0 else float('inf')
    P_spectrum = 0.023 * r0**(-5/3) * (K_freq**2 + k0_vk**2)**(-11/6) * torch.exp(-(K_freq**2)/(km_vk**2))
    P_spectrum[0, 0] = 0
    random_field = torch.randn(size, size, device=device) + 1j * torch.randn(size, size, device=device)
    phase_screen = torch.fft.ifft2((1 / screen_size_val) * torch.sqrt(P_spectrum) * random_field).real * scale_factor
    return phase_screen


def compute_grid(h: int, w: int, device: torch.device):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).float()
    norm_tensor = torch.tensor([w-1, h-1], dtype=torch.float32, device=device)
    return grid, norm_tensor


def nollToZernInd(j):
    if not isinstance(j, int) or j <= 0:
        raise ValueError("Noll index j must be a positive integer.")
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m_abs = (n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0)
    return n, (-1)**(j % 2) * m_abs


_gamma_cache: dict = {}


def gamma_pytorch(z_input):
    global _gamma_cache
    is_tensor = isinstance(z_input, torch.Tensor)
    if is_tensor:
        original_dtype = z_input.dtype
        z_64 = z_input.to(torch.float64)
        current_device = z_input.device
        z_np = z_64.cpu().numpy()
    else:
        if z_input in _gamma_cache:
            return _gamma_cache[z_input]
        z_np = float(z_input)
        current_device = torch.device("cpu")

    if is_tensor:
        result = torch.zeros_like(z_64)
        mask_lt_half = z_64 < 0.5
        if torch.any(mask_lt_half):
            z_lt = z_64[mask_lt_half]
            g1mz = gamma_pytorch(1.0 - z_lt)
            result[mask_lt_half] = math.pi / (torch.sin(math.pi * z_lt) * g1mz.to(current_device))
        mask_ge = ~mask_lt_half
        if torch.any(mask_ge):
            result[mask_ge] = torch.tensor(
                scipy_gamma(z_np[mask_ge.cpu().numpy()]), dtype=torch.float64, device=current_device)
        return result.to(original_dtype)
    else:
        if z_np < 0.5:
            result_scalar = math.pi / (math.sin(math.pi * z_np) * gamma_pytorch(1.0 - z_np))
        else:
            result_scalar = scipy_gamma(z_np)
        _gamma_cache[z_input] = result_scalar
        return result_scalar


_noll_cov_cache: dict = {}


def nollCovMat_pytorch(num_zern, D_over_r0, L0_val=float('inf'), l0_val=0.0,
                        D_aperture_val=None, wavelength_val=0.525e-6,
                        L_prop_val=None, device_val=None):
    if device_val is None:
        device_val = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if L0_val <= l0_val:
        raise ValueError("Outer scale L0 must be greater than inner scale l0.")
    spectrum_type = "Kolmogorov" if L0_val == float('inf') and l0_val == 0.0 else "Von Karman"
    if spectrum_type == "Von Karman" and (D_aperture_val is None or L_prop_val is None):
        raise ValueError("D_aperture and L_prop are required for Von Karman.")

    D_over_r0 = float(D_over_r0)
    cache_key = (num_zern, D_over_r0, L0_val, l0_val, D_aperture_val, wavelength_val, L_prop_val, str(device_val))
    if cache_key in _noll_cov_cache:
        return _noll_cov_cache[cache_key]

    C_mat = torch.zeros((num_zern, num_zern), dtype=torch.float64, device=device_val)
    g14_3 = gamma_pytorch(torch.tensor(14.0 / 3.0, dtype=torch.float64, device=device_val))
    kolmogorov_factor = (D_over_r0 ** (5.0 / 3.0) *
                         (gamma_pytorch(torch.tensor(11.0/6.0, dtype=torch.float64, device=device_val))**2 *
                          (24.0/5.0 * gamma_pytorch(torch.tensor(6.0/5.0, dtype=torch.float64, device=device_val)))**(5.0/6.0)) /
                         (2.0 * math.pi**(2.0/3.0)))

    k_wave = 2 * math.pi / wavelength_val
    r0_vk = None
    Cn_sq = None
    if spectrum_type == "Von Karman":
        r0_vk = D_aperture_val / D_over_r0
        Cn_sq = torch.tensor((r0_vk**(-5.0/3.0)) / (0.423 * k_wave**2 * L_prop_val),
                              device=device_val, dtype=torch.float64)

    def zernike_filter(n_in, m_in, D_in, kappa_grid, dev):
        kappa_64 = kappa_grid.to(torch.float64)
        arg = torch.pi * D_in * kappa_64
        safe_arg = torch.where(arg == 0, torch.tensor(1.0, device=dev, dtype=torch.float64), arg)
        bessel_vals = torch.tensor(sp.jv(n_in + 1, safe_arg.cpu().numpy()), device=dev, dtype=torch.float64)
        filt = torch.sqrt(torch.tensor(n_in + 1, dtype=torch.float64, device=dev)) * (2 * bessel_vals / safe_arg)
        limit = torch.sqrt(torch.tensor(n_in + 1, dtype=torch.float64, device=dev)) * (1.0 if n_in == 0 else 0.0)
        return torch.where(arg == 0, limit, filt)

    def phi_vk(kappa, Cn_sq_val, L0, l0, dev):
        k64 = kappa.to(torch.float64)
        k0_sq = torch.tensor((2.0 * math.pi / L0)**2 if L0 != float('inf') else 0.0, device=dev, dtype=torch.float64)
        psd = 0.033 * Cn_sq_val * (k64**2 + k0_sq)**(-11.0/6.0)
        if l0 > 0:
            psd *= torch.exp(-(k64 / (5.92 / l0))**2)
        return psd

    for i_idx in range(num_zern):
        for j_idx in range(num_zern):
            j_i, j_j = i_idx + 2, j_idx + 2
            ni, mi = nollToZernInd(j_i)
            nj, mj = nollToZernInd(j_j)
            mi_int, mj_int = int(mi), int(mj)
            if mi_int == mj_int and (ni + nj) % 2 == 0:
                phase_term = (-1.0)**((ni + nj - 2 * abs(mi_int)) / 2.0)
                term1 = torch.sqrt(torch.tensor((ni + 1) * (nj + 1), dtype=torch.float64, device=device_val))
                if spectrum_type == "Von Karman":
                    try:
                        kmin = 1e-3 / L0_val if L0_val != float('inf') else 1e-4
                        kmax = 10.0 / l0_val if l0_val > 0 else 1e4
                        kappa_grid = torch.logspace(math.log10(kmin), math.log10(kmax), 1000,
                                                     device=device_val, dtype=torch.float64)
                        Fi = zernike_filter(ni, mi_int, D_aperture_val, kappa_grid, device_val)
                        Fj = zernike_filter(nj, mj_int, D_aperture_val, kappa_grid, device_val)
                        psd = phi_vk(kappa_grid, Cn_sq, L0_val, l0_val, device_val)
                        integrand = 2.0 * math.pi * kappa_grid * psd * Fi * Fj
                        C_mat[i_idx, j_idx] = phase_term * torch.trapz(integrand, kappa_grid) * k_wave**2 * L_prop_val
                    except Exception:
                        C_mat[i_idx, j_idx] = 0.0
                else:
                    try:
                        arg_num = (ni + nj - 5.0/3.0) / 2.0
                        arg_den1 = (-ni + nj + 17.0/3.0) / 2.0
                        arg_den2 = (ni - nj + 17.0/3.0) / 2.0
                        arg_den3 = (ni + nj + 23.0/3.0) / 2.0
                        gammas = [gamma_pytorch(torch.tensor(a, dtype=torch.float64, device=device_val))
                                  for a in (arg_num, arg_den1, arg_den2, arg_den3)]
                        if all(torch.isfinite(g) for g in gammas):
                            den = gammas[1] * gammas[2] * gammas[3]
                            dim_int = 0.0 if abs(den.item()) < 1e-15 else (g14_3 * gammas[0] / den)
                            C_mat[i_idx, j_idx] = kolmogorov_factor * phase_term * term1 * dim_int
                    except Exception:
                        C_mat[i_idx, j_idx] = 0.0

    result = C_mat.float()
    _noll_cov_cache[cache_key] = result
    return result


_noll_decomp_cache: dict = {}


def genZernikeCoeff_correlated(num_zern, D_over_r0, L0_val=float('inf'), l0_val=0.0,
                                D_aperture_val=None, wavelength_val=0.525e-6,
                                L_prop_val=None, batch_size=1, device_val=None):
    if device_val is None:
        device_val = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_over_r0 = float(D_over_r0)
    decomp_key = (num_zern, D_over_r0, L0_val, l0_val, D_aperture_val, wavelength_val, L_prop_val, str(device_val))

    if decomp_key not in _noll_decomp_cache:
        C_cov = nollCovMat_pytorch(num_zern, D_over_r0, L0_val, l0_val,
                                    D_aperture_val, wavelength_val, L_prop_val, device_val)
        if not torch.all(torch.isfinite(C_cov)):
            C_cov = torch.nan_to_num(C_cov, nan=0.0, posinf=1e5, neginf=-1e5)
        eigenvalues, eigenvectors = torch.linalg.eigh(C_cov)
        eigenvalues = torch.clamp(eigenvalues, min=1e-7)
        R = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
        if torch.isnan(R).any():
            raise RuntimeError("NaNs in Noll covariance decomposition.")
        _noll_decomp_cache[decomp_key] = R

    R = _noll_decomp_cache[decomp_key].to(torch.float32)
    noise = torch.randn(num_zern, batch_size, dtype=torch.float32, device=device_val)
    return torch.matmul(R, noise).T


def torch_fft_convolve2d_rfft_blend(input_tensor, kernel_ffts, H_out, W_out, K_kernel, centroid_shifts):
    _, C_img, H_img, W_img = input_tensor.shape
    N_levels, _, H_fft, W_fft_half = kernel_ffts.shape
    fft_shape = [H_fft, (W_fft_half - 1) * 2]
    input_padded = F.pad(input_tensor, (0, fft_shape[1] - W_img, 0, fft_shape[0] - H_img))
    input_fft = torch.fft.rfft2(input_padded.float(), s=fft_shape, dim=(-2, -1)).to(torch.complex64)
    fy = torch.fft.fftfreq(fft_shape[0], device=kernel_ffts.device).reshape(1, 1, -1, 1).expand(N_levels, 1, -1, W_fft_half)
    fx = torch.fft.rfftfreq(fft_shape[1], device=kernel_ffts.device).reshape(1, 1, 1, -1).expand(N_levels, 1, H_fft, -1)
    shift_y = centroid_shifts[:, 0].reshape(-1, 1, 1, 1)
    shift_x = centroid_shifts[:, 1].reshape(-1, 1, 1, 1)
    phase_ramp = torch.exp(1j * (-2.0 * math.pi * (fy * shift_y + fx * shift_x)).to(kernel_ffts.dtype))
    output_ifft = torch.fft.irfft2(input_fft * (kernel_ffts * phase_ramp), s=fft_shape, dim=(-2, -1))
    output_ifft = output_ifft.to(input_tensor.dtype)
    s_y, s_x = K_kernel // 2, K_kernel // 2
    cropped = output_ifft[..., s_y:s_y + H_out, s_x:s_x + W_out]
    if cropped.shape[-2] != H_out or cropped.shape[-1] != W_out:
        ch, cw = fft_shape[0] // 2, fft_shape[1] // 2
        cropped = output_ifft[..., ch - H_out//2:ch - H_out//2 + H_out, cw - W_out//2:cw - W_out//2 + W_out]
    return cropped


def precompute_blur_data(img_shape, patch_size_val, num_patches, device_val):
    H_img, W_img = img_shape
    total_patches = num_patches ** 2
    stride = max(1, H_img // num_patches)
    patch_centers_h = torch.arange(stride // 2, H_img, stride, device=device_val)
    patch_centers_w = torch.arange(stride // 2, W_img, stride, device=device_val)
    if len(patch_centers_h) != num_patches:
        patch_centers_h = torch.linspace(stride // 2, H_img - stride // 2, num_patches, device=device_val)
    if len(patch_centers_w) != num_patches:
        patch_centers_w = torch.linspace(stride // 2, W_img - stride // 2, num_patches, device=device_val)
    yy, xx = torch.meshgrid(patch_centers_h, patch_centers_w, indexing='ij')
    patch_coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
    sigma_gauss = stride / 2.0
    r = math.ceil(sigma_gauss * 2)
    gs = int(r * 2) + 1
    x_g = torch.linspace(-r, r, gs, device=device_val)
    X_g, Y_g = torch.meshgrid(x_g, x_g, indexing='ij')
    gauss_kernel = torch.exp(-(X_g**2 + Y_g**2) / (2 * sigma_gauss**2)).unsqueeze(0).unsqueeze(0)
    gauss_kernel /= gauss_kernel.sum()
    masks = torch.zeros(total_patches, H_img, W_img, device=device_val)
    coords_h = patch_coords[:, 0].round().long().clamp(0, H_img - 1)
    coords_w = patch_coords[:, 1].round().long().clamp(0, W_img - 1)
    masks[torch.arange(total_patches), coords_h, coords_w] = 1.0
    masks = masks.unsqueeze(1)
    K_g = gauss_kernel.shape[-1]
    fft_shape_g = [H_img + K_g - 1, W_img + K_g - 1]
    kfft = torch.fft.rfft2(
        F.pad(gauss_kernel, (0, fft_shape_g[1] - K_g, 0, fft_shape_g[0] - K_g)),
        s=fft_shape_g, dim=(-2, -1)).expand(total_patches, -1, -1, -1)
    zero_shifts = torch.zeros(total_patches, 2, device=device_val)
    smoothed = torch_fft_convolve2d_rfft_blend(masks.float(), kfft, H_img, W_img, K_g, zero_shifts).squeeze(0)
    smoothed = smoothed / (smoothed.sum(dim=0, keepdim=True) + 1e-9)
    return smoothed


def fast_blur_image_single_optimized(image, depth_map, smoothed_masks, coeffs_ref_patches_input,
                                      patch_size_arg, scaling_arg, nn_base_arg,
                                      num_psf_levels=20, num_zern_arg=5,
                                      min_Dr0_arg=0.4, max_Dr0_arg=4.0,
                                      L0_arg=float('inf'), l0_arg=0.0,
                                      D_aperture_arg=0.2, wavelength_arg=0.525e-6,
                                      useDepth_arg=True, L_prop_arg=7000.0):
    device_val = image.device
    if image.dim() == 3:
        C_img, H_img, W_img = image.shape
    elif image.dim() == 2:
        H_img, W_img = image.shape
        C_img = 1
        image = image.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    image_batched = image.unsqueeze(0).half()
    P_patches = smoothed_masks.shape[0]
    K_conv = max(1, round(nn_base_arg / scaling_arg))

    Dr0_ref = (min_Dr0_arg + max_Dr0_arg) / 2.0
    Dr0_levels = (torch.linspace(min_Dr0_arg, max_Dr0_arg, num_psf_levels, device=device_val)
                  if num_psf_levels > 1
                  else torch.tensor([Dr0_ref], device=device_val))
    scaling_factors = (Dr0_levels / Dr0_ref) ** (5.0 / 6.0)
    coeffs_levels = coeffs_ref_patches_input.unsqueeze(1) * scaling_factors.reshape(1, -1, 1)
    coeffs_batch = coeffs_levels.reshape(-1, num_zern_arg)

    psf_raw, centroid_shifts = psfGen(nn_base_arg, coeffs_batch, wavelength=wavelength_arg, D_aper=D_aperture_arg)
    psf_scaled = custom_resize(psf_raw.unsqueeze(1), (K_conv, K_conv)).squeeze(1)
    psf_scaled = psf_scaled / (psf_scaled.sum(dim=(-2, -1), keepdim=True) + 1e-9)
    centroid_shifts = centroid_shifts * (K_conv / nn_base_arg)

    fft_shape = [H_img + K_conv - 1, W_img + K_conv - 1]
    kernel_padded = F.pad(psf_scaled.unsqueeze(1).half(),
                           (0, fft_shape[1] - K_conv, 0, fft_shape[0] - K_conv))
    kernel_ffts = torch.fft.rfft2(kernel_padded.float(), s=fft_shape, dim=(-2, -1))
    blurred_batch = torch_fft_convolve2d_rfft_blend(
        image_batched, kernel_ffts, H_img, W_img, K_conv, centroid_shifts)
    blurred_levels_patches = blurred_batch.reshape(P_patches, num_psf_levels, C_img, H_img, W_img)

    if useDepth_arg:
        Dr0_map = torch.clamp(min_Dr0_arg + depth_map * (max_Dr0_arg - min_Dr0_arg),
                               min_Dr0_arg, max_Dr0_arg)
    else:
        Dr0_map = torch.full_like(depth_map, Dr0_ref)

    if num_psf_levels <= 1:
        blended = blurred_levels_patches[:, 0]
    else:
        Dr0_norm = (Dr0_map - min_Dr0_arg) / (max_Dr0_arg - min_Dr0_arg + 1e-9)
        frac_idx = Dr0_norm * (num_psf_levels - 1)
        idx0 = torch.floor(frac_idx).long().clamp(0, num_psf_levels - 2)
        idx1 = idx0 + 1
        w = (frac_idx - idx0.float()).unsqueeze(0).unsqueeze(0)
        idx0_exp = idx0.reshape(1, 1, H_img, W_img).expand(P_patches, C_img, -1, -1).unsqueeze(1)
        idx1_exp = idx1.reshape(1, 1, H_img, W_img).expand(P_patches, C_img, -1, -1).unsqueeze(1)
        val0 = torch.gather(blurred_levels_patches, 1, idx0_exp).squeeze(1)
        val1 = torch.gather(blurred_levels_patches, 1, idx1_exp).squeeze(1)
        blended = (1.0 - w.half()) * val0 + w.half() * val1

    final = (blended.float() * smoothed_masks).sum(dim=0)
    if C_img == 1 and final.dim() == 3:
        final = final.squeeze(0)
    return final.float()


# ---------------------------------------------------------------------------
# DAATSimPerturber
# ---------------------------------------------------------------------------

class DAATSimPerturber:
    """
    Apply single-frame atmospheric turbulence to a uint8 HxWx3 RGB numpy image.

    Args:
        r0:     Fried parameter in metres. Lower = stronger turbulence (0.02–0.1 typical).
        cfg:    Parameter namespace. Defaults to the daatsim_config module.
        device: torch device string or object. Auto-detected if None.
    """

    def __init__(self, r0: float, cfg=None, device=None, use_depth: bool = None,
                 L_prop: float = None, L0: float = None, depth_fusion=None,
                 alpha_tilt: float = None, alpha_blur: float = None):
        self.cfg = cfg or _default_config
        self.r0 = float(r0)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Physical parameters — constructor args override config values when provided
        self.size_d = self.cfg.SIZE_D_PARAM
        self.num_zern = self.cfg.NUM_ZERN_PARAM
        self.D_aperture = self.cfg.D_APERTURE_PARAM
        self.L_prop = L_prop if L_prop is not None else self.cfg.L_PROP_PARAM
        self.wavelength = self.cfg.WAVELENGTH_PARAM
        self.obj_size = self.cfg.OBJ_SIZE_PARAM
        self.L0 = L0 if L0 is not None else self.cfg.L0_PARAM
        self.l0 = self.cfg.l0_PARAM
        self.focal_length = self.cfg.FOCAL_LENGTH_PARAM
        self.phase_screen_scale = self.cfg.PHASE_SCREEN_SCALE_FACTOR_PARAM
        self.tilt_scale = self.cfg.TILT_SCALE_FACTOR_PARAM
        self.NN_base = self.cfg.NN_BASE_PARAM
        self.use_depth = use_depth if use_depth is not None else self.cfg.USE_DEPTH_PARAM
        self.use_correlated = self.cfg.USE_CORRELATED_COEFFS_PARAM

        # Derived
        delta0 = self.L_prop * self.wavelength / (2 * self.D_aperture)
        self.scaling = self.obj_size / (self.size_d * delta0)
        Dr0 = self.D_aperture / self.r0
        self.min_Dr0 = max(1e-4, Dr0 * self.cfg.MIN_BLUR_REL_STRENGTH_PARAM)
        self.max_Dr0 = Dr0 * self.cfg.MAX_BLUR_REL_STRENGTH_PARAM
        if self.min_Dr0 >= self.max_Dr0:
            self.min_Dr0 = self.max_Dr0 * 0.5

        self.num_psf_levels = (self.cfg.NUM_PSF_LEVELS_HIGH_RES if self.size_d <= 512
                               else self.cfg.NUM_PSF_LEVELS_LOW_RES)

        # Adaptive patch layout
        smax = (delta0 / self.D_aperture * self.size_d) * self.scaling
        temp = np.arange(1, 101)
        patchN = temp[np.argmin((smax / temp - 2)**2)]
        self.patch_size = max(1, min(round(self.size_d / patchN), self.size_d))
        self.num_patches = max(1, self.size_d // self.patch_size)

        img_size = (self.size_d, self.size_d)
        self.smoothed_masks = precompute_blur_data(img_size, self.patch_size, self.num_patches, self.device)
        self._grid, self._norm = compute_grid(self.size_d, self.size_d, self.device)

        # An explicitly supplied depth pipeline can provide spatial depth variation.
        if depth_fusion is not None:
            self.depth_fusion = depth_fusion
        elif self.use_depth:
            raise ValueError("Without a depth pipeline, DAATSim requires use_depth=False")
        else:
            self.depth_fusion = None

        self.alpha_tilt = self.cfg.ALPHA_TILT_PARAM if alpha_tilt is None else alpha_tilt
        self.alpha_blur = self.cfg.ALPHA_BLUR_PARAM if alpha_blur is None else alpha_blur
        # AR(1) temporal state — None until the first frame is processed
        self._phase_screen_state: torch.Tensor | None = None
        self._coeffs_state: torch.Tensor | None = None

        print(f"DAATSimPerturber ready — r0={self.r0}, L_prop={self.L_prop}, L0={self.L0}, "
              f"device={self.device}, patches={self.num_patches}x{self.num_patches}, "
              f"alpha_tilt={self.alpha_tilt}, alpha_blur={self.alpha_blur}")

    def perturb(self, image: np.ndarray) -> np.ndarray:
        """
        Apply atmospheric turbulence to a uint8 HxWx3 RGB numpy image.

        Input is resized to SIZE_D_PARAM for simulation; output is at that resolution.
        Returns uint8 HxWx3 RGB numpy array.
        """
        img_size = (self.size_d, self.size_d)
        pil_img = Image.fromarray(image).resize(img_size)

        if self.use_depth and self.depth_fusion is not None:
            depth_metric = self.depth_fusion.process_pil_image(pil_img, target_size=img_size, save=False)
            depth_t = torch.tensor(depth_metric, dtype=torch.float32, device=self.device)
            depth_norm = depth_t / (depth_t.max() + 1e-9)
        else:
            depth_norm = torch.zeros(self.size_d, self.size_d, device=self.device)

        img_tensor = torch.from_numpy(
            np.array(pil_img).transpose(2, 0, 1).astype(np.float32) / 255.0
        ).to(self.device)

        # Tilt warp via phase screen gradient
        phase_screen = generate_phase_screen(
            self.size_d, self.r0, self.L0, self.l0,
            self.obj_size, self.device, self.phase_screen_scale)
        if self._phase_screen_state is not None:
            phase_screen = (self.alpha_tilt * self._phase_screen_state
                            + (1.0 - self.alpha_tilt) * phase_screen)
        self._phase_screen_state = phase_screen.detach()
        dy, dx = torch.gradient(phase_screen)
        angle_x = dx * self.wavelength / (2 * torch.pi)
        angle_y = dy * self.wavelength / (2 * torch.pi)
        depth_mul = depth_norm if self.use_depth else torch.ones_like(depth_norm)
        img_scale = (self.size_d / 128) ** 1.8
        ndx = angle_x * self.focal_length * 2.0 / self.obj_size * self.tilt_scale * depth_mul * img_scale
        ndy = angle_y * self.focal_length * 2.0 / self.obj_size * self.tilt_scale * depth_mul * img_scale
        grid_norm = (self._grid / self._norm.view(1, 1, 2)) * 2.0 - 1.0
        disp_grid = torch.clamp(grid_norm + torch.stack([ndx, ndy], dim=-1), -1.0, 1.0)
        tilted = F.grid_sample(
            img_tensor.unsqueeze(0), disp_grid.unsqueeze(0),
            mode='bilinear', padding_mode='reflection', align_corners=True)[0]

        # Spatially-varying blur via Zernike PSFs
        P = self.num_patches ** 2
        Dr0_base = (self.min_Dr0 + self.max_Dr0) / 2.0
        if self.use_correlated:
            coeffs = genZernikeCoeff_correlated(
                self.num_zern, Dr0_base, L0_val=self.L0, l0_val=self.l0,
                D_aperture_val=self.D_aperture, wavelength_val=self.wavelength,
                L_prop_val=self.L_prop, batch_size=P, device_val=self.device)
        else:
            coeffs = genZernikeCoeff_independent(
                self.num_zern, Dr0_base, batch_size=P, device=self.device)
        if self._coeffs_state is not None:
            coeffs = (self.alpha_blur * self._coeffs_state
                      + (1.0 - self.alpha_blur) * coeffs)
        self._coeffs_state = coeffs.detach()

        with torch.no_grad():
            blurred = fast_blur_image_single_optimized(
                image=tilted,
                depth_map=depth_norm,
                smoothed_masks=self.smoothed_masks,
                coeffs_ref_patches_input=coeffs,
                patch_size_arg=self.patch_size,
                scaling_arg=self.scaling,
                nn_base_arg=self.NN_base,
                num_psf_levels=self.num_psf_levels,
                num_zern_arg=self.num_zern,
                min_Dr0_arg=self.min_Dr0,
                max_Dr0_arg=self.max_Dr0,
                L0_arg=self.L0,
                l0_arg=self.l0,
                D_aperture_arg=self.D_aperture,
                wavelength_arg=self.wavelength,
                useDepth_arg=self.use_depth,
                L_prop_arg=self.L_prop,
            )

        out = blurred.permute(1, 2, 0).cpu().numpy() if blurred.dim() == 3 else blurred.cpu().numpy()
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """Clear AR(1) temporal state. Call between independent sequences."""
        self._phase_screen_state = None
        self._coeffs_state = None
