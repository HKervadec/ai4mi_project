
from dataclasses import dataclass
import math, random
import torch
import torch.nn.functional as F

@dataclass
class AugConfig2D:
    # --- Rotation controls (kept mild for small tubular organs) ---
    p_rotate_cont: float = 0.8
    rot_deg: float = 10.0          # ↓ from 12 to 10: more conservative
    p_rot90: float = 0.15          # ↓ lower prob. to reduce discontinuities

    # --- Affine (scale / shear / translate / flip) ---
    scale_min: float = 0.95        # narrower scale range to reduce distortion
    scale_max: float = 1.08
    shear_deg: float = 6.0         # ↓ from 8 to 6
    translate: float = 0.02        # ↓ from 0.03 to 0.02 (as a fraction of H/W)
    p_flip: float = 0.5

    # --- Intensity (image only) ---
    p_gamma: float = 0.3
    gamma_min: float = 0.88
    gamma_max: float = 1.12
    p_noise: float = 0.3
    noise_std: float = 0.02
    p_bias_field: float = 0.25
    bias_strength: float = 0.12
    bias_downscale: int = 32

    # --- ROI-aware small-object focus ---
    # Which one-hot channels are "small targets" (e.g., esophagus). With a given probability,
    # we re-center the crop around foreground to keep small targets in view.
    small_class_indices: tuple = (0, 2)   # change if your esophagus is not channel 0
    p_roi_focus: float = 0.35             # prob. to apply ROI-centered recentring
    roi_jitter_frac: float = 0.10         # random center jitter (fraction of H/W)
    min_fg_pixels: int = 24               # skip ROI focus if too few foreground pixels

    # --- Elastic deformation ( mild & safe) ---
    p_elastic: float = 0.25
    elastic_alpha: float = 1.5            # displacement magnitude in pixels (mild)
    elastic_sigma: float = 6.0            # Gaussian smoothing sigma (larger = smoother)

def _affine_grid(theta, size):
    return F.affine_grid(theta, size=size, align_corners=False)

def _apply_grid(x, grid, mode):
    return F.grid_sample(x, grid, mode=mode, padding_mode="border", align_corners=False)

def _gaussian_kernel1d(sigma: float, radius: int):
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(coords**2) / (2 * sigma * sigma))
    k /= k.sum()
    return k

def _gaussian_blur(img: torch.Tensor, sigma: float):
    """Separable 2D Gaussian blur used for bias-field and elastic smoothing."""
    radius = max(1, int(3 * sigma))
    k1 = _gaussian_kernel1d(sigma, radius).to(img.device, img.dtype)
    ky = k1.view(1,1,-1,1); kx = k1.view(1,1,1,-1)
    C = img.shape[1]
    img = F.conv2d(img, ky.expand(C,1,-1,1), padding=(radius,0), groups=C)
    img = F.conv2d(img, kx.expand(C,1,1,-1), padding=(0,radius), groups=C)
    return img

def _make_elastic_grid(H, W, device, dtype, alpha=1.5, sigma=6.0):
    """Build a mild elastic deformation grid in normalized device coordinates ([-1, 1])."""
    # Base grid in NDC
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    base = torch.stack([xx, yy], dim=-1)[None, ...]  # [1, H, W, 2]

    # Random displacement in pixel space → smooth → convert to NDC
    disp = torch.randn(1, 2, H, W, device=device, dtype=dtype)
    disp = _gaussian_blur(disp, sigma=max(1.0, sigma/3.0))
    # pixel → NDC: dx_ndc = 2*dx/(W-1), dy_ndc = 2*dy/(H-1)
    scale_x = 2.0 / max(1.0, (W - 1))
    scale_y = 2.0 / max(1.0, (H - 1))
    disp[:, 0] = disp[:, 0] * alpha * scale_x
    disp[:, 1] = disp[:, 1] * alpha * scale_y
    disp = disp.permute(0, 2, 3, 1)  # [1, H, W, 2]
    return (base + disp).clamp(-1, 1)

def _center_crop_around_foreground(img_b, gt_b, focus_chs, jitter_frac=0.1, min_fg=24):
    """
    Re-center the field-of-view around the centroid of a small-object foreground channel,
    without changing spatial resolution. Implemented as a pure translation via an affine grid.
    Returns (img_c, gt_c) or None if no suitable foreground is found.
    """
    _, _, H, W = img_b.shape
    device, dtype = img_b.device, img_b.dtype

    # Pick the first small-class that actually has foreground
    chosen = None
    for ch in focus_chs:
        fg = (gt_b[:, ch:ch+1] > 0.5).float()  # [B,1,H,W]
        if fg.sum() >= min_fg:
            chosen = ch
            break
    if chosen is None:
        return None

    fg = (gt_b[:, chosen:chosen+1] > 0.5).float()
    # Foreground centroid
    with torch.no_grad():
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        m = fg[0, 0]
        s = m.sum()
        if s < min_fg:
            return None
        cy = (m * yy).sum() / (s + 1e-6)
        cx = (m * xx).sum() / (s + 1e-6)

    # Small random jitter around the centroid
    jx = (random.uniform(-jitter_frac, jitter_frac)) * W
    jy = (random.uniform(-jitter_frac, jitter_frac)) * H
    cx = (cx + jx).clamp(0, W - 1)
    cy = (cy + jy).clamp(0, H - 1)

    # Translate (cx, cy) to image center (W/2, H/2) → compute NDC translation
    tx_pix = (W - 1)/2.0 - cx
    ty_pix = (H - 1)/2.0 - cy
    tx_ndc = 2.0 * tx_pix / max(1.0, (W - 1))
    ty_ndc = 2.0 * ty_pix / max(1.0, (H - 1))

    # Pure-translation affine matrix
    A = torch.tensor([[1.0, 0.0, tx_ndc],
                      [0.0, 1.0, ty_ndc]], device=device, dtype=dtype).unsqueeze(0)

    grid = F.affine_grid(A, size=img_b.size(), align_corners=False)
    img_c = _apply_grid(img_b, grid, mode="bilinear")
    gt_c  = _apply_grid(gt_b,  grid, mode="nearest")
    return img_c, gt_c

class OnlineAugment2D:
    """
    Apply synchronized geometry to image & mask; intensity changes to image only.
    img: [C,H,W] in [0,1]; gt_onehot: [K,H,W] float one-hot
    returns (img_aug, gt_aug)
    """
    def __init__(self, cfg: AugConfig2D = AugConfig2D()):
        self.cfg = cfg

    def __call__(self, img: torch.Tensor, gt_onehot: torch.Tensor):
        C, H, W = img.shape
        K = gt_onehot.shape[0]
        device, dtype = img.device, img.dtype

        img_b = img.unsqueeze(0)    # [1, C, H, W]
        gt_b  = gt_onehot.unsqueeze(0).to(device=img.device, dtype=img.dtype)

        # ===== 0) ROI-aware re-centering (prioritize small targets) =====
        if random.random() < self.cfg.p_roi_focus and len(self.cfg.small_class_indices) > 0:
            out = _center_crop_around_foreground(
                img_b, gt_b,
                focus_chs=tuple(i for i in self.cfg.small_class_indices if 0 <= i < K),
                jitter_frac=self.cfg.roi_jitter_frac,
                min_fg=self.cfg.min_fg_pixels
            )
            if out is not None:
                img_b, gt_b = out  # small object moved towards the image center

        # ===== 1) Continuous random rotation / general affine =====
        if random.random() < self.cfg.p_rotate_cont:
            ang = math.radians(random.uniform(-self.cfg.rot_deg, self.cfg.rot_deg))
        else:
            ang = 0.0

        sc  = random.uniform(self.cfg.scale_min, self.cfg.scale_max)
        sh  = math.tan(math.radians(random.uniform(-self.cfg.shear_deg, self.cfg.shear_deg)))
        tx  = random.uniform(self.cfg.translate * -1, self.cfg.translate) * 2.0  # NDC
        ty  = random.uniform(self.cfg.translate * -1, self.cfg.translate) * 2.0

        ca, sa = math.cos(ang), math.sin(ang)
        A = torch.tensor([[sc*ca, -sc*sa + sh*sc*ca, tx],
                          [sc*sa,  sc*ca + sh*sc*sa, ty]], dtype=dtype, device=device).unsqueeze(0)

        # Apply geometry (image: bilinear, mask: nearest)
        grid = _affine_grid(A, size=img_b.size())
        img_geo = _apply_grid(img_b, grid, mode="bilinear")
        gt_geo  = _apply_grid(gt_b,  grid, mode="nearest")

        # Optional horizontal flip
        if random.random() < self.cfg.p_flip:
            img_geo = torch.flip(img_geo, dims=[-1])
            gt_geo  = torch.flip(gt_geo,  dims=[-1])

        # Additional 90° * k rotation (reduced probability)
        if random.random() < self.cfg.p_rot90:
            k = random.choice([1, 2, 3])
            img_geo = torch.rot90(img_geo, k=k, dims=(-2, -1))
            gt_geo  = torch.rot90(gt_geo,  k=k, dims=(-2, -1))

        # ===== 1.5) Mild elastic deformation =====
        if random.random() < self.cfg.p_elastic:
            egrid = _make_elastic_grid(H, W, device, dtype,
                                       alpha=self.cfg.elastic_alpha,
                                       sigma=self.cfg.elastic_sigma)
            img_geo = _apply_grid(img_geo, egrid, mode="bilinear")
            gt_geo  = _apply_grid(gt_geo,  egrid, mode="nearest")

        # Keep mask as clean one-hot
        gt_geo = (gt_geo > 0.5).float()
        img_geo = img_geo.clamp_(0, 1)

        # ===== 2) Image-only intensity transforms =====
        if random.random() < self.cfg.p_gamma:
            g = random.uniform(self.cfg.gamma_min, self.cfg.gamma_max)
            img_geo = img_geo.clamp_(1e-6, 1).pow(g)

        if random.random() < self.cfg.p_noise:
            img_geo = (img_geo + torch.randn_like(img_geo)*self.cfg.noise_std).clamp_(0, 1)

        if random.random() < self.cfg.p_bias_field:
            ds = self.cfg.bias_downscale
            h, w = max(2, H//ds), max(2, W//ds)
            low = torch.randn(1,1,h,w, device=device, dtype=dtype)
            low = _gaussian_blur(low, sigma=1.0)
            low = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
            low = (low - low.mean()) / (low.std() + 1e-6)
            field = 1.0 + self.cfg.bias_strength * low
            img_geo = (img_geo * field).clamp_(0, 1)

        return img_geo.squeeze(0), gt_geo.squeeze(0)
