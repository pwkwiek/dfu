"""
predict_unet_lab_eri.py

Load your saved checkpoint (unet_lab_eri_best.pt), run inference on images in a
test folder, and save predicted masks (plus optional overlays).

Usage (example):
    python predict_unet_lab_eri.py \
        --ckpt unet_lab_eri_best.pt \
        --test_dir /path/to/test_images \
        --out_dir /path/to/output \
        --thr 0.5 \
        --size 512 512 \
        --save_overlay
"""

import os
import glob
import argparse
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.color import rgb2lab


# -----------------------------
# 1) Same preprocessing as training
# -----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def rgb_uint8_to_lab_eri(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    rgb_uint8: HxWx3 uint8 [0..255]
    returns HxWx4 float32 with channels [L,a,b,eri]
      L in [0,1]
      a,b approx [-1,1]
      eri clipped [-1,1]
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    eri = R - 0.5 * (G + B)
    eri = np.clip(eri, -1.0, 1.0)

    lab = rgb2lab(rgb)  # L: [0..100], a/b about [-128..127]
    L = lab[..., 0] / 100.0
    a = lab[..., 1] / 128.0
    b = lab[..., 2] / 128.0

    x = np.stack([L, a, b, eri], axis=-1).astype(np.float32)
    return x


def normalize_per_channel(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, None, :]) / (std[None, None, :] + 1e-8)


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def list_images(folder: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


# -----------------------------
# 2) Model definition (same as training)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# -----------------------------
# 3) Load checkpoint
# -----------------------------
def load_checkpoint_model(
    ckpt_path: str,
    device: Optional[str] = None,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, dict, str]:
    """
    Returns: (model, mean, std, cfg_dict, device_used)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    base = int(cfg.get("base_channels", 32))

    model = UNet(in_channels=4, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    return model, mean, std, cfg, device


# -----------------------------
# 4) Inference utilities
# -----------------------------
@torch.no_grad()
def predict_mask_single(
    model: nn.Module,
    rgb_uint8: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    size: Tuple[int, int] = (512, 512),
    device: str = "cpu",
    thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      prob: (H,W) float32 in [0,1]
      mask: (H,W) uint8 in {0,255}
    """
    H, W = size
    rgb_resized = cv2.resize(rgb_uint8, (W, H), interpolation=cv2.INTER_LINEAR)

    x = rgb_uint8_to_lab_eri(rgb_resized)          # HxWx4
    x = normalize_per_channel(x, mean, std)        # HxWx4
    x_t = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).float().to(device)  # 1x4xHxW

    logits = model(x_t)                            # 1x1xHxW
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    mask = (prob >= thr).astype(np.uint8) * 255
    return prob, mask


def overlay_mask(rgb_uint8: np.ndarray, mask_255: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Simple red overlay (no matplotlib needed). Input RGB, output RGB.
    """
    rgb = rgb_uint8.copy()
    if mask_255.ndim == 3:
        mask_255 = mask_255[..., 0]
    m = (mask_255 > 127)

    overlay = rgb.copy()
    overlay[m] = np.clip((1 - alpha) * overlay[m] + alpha * np.array([255, 0, 0], dtype=np.float32), 0, 255)
    return overlay.astype(np.uint8)


# -----------------------------
# 5) Batch prediction from folder
# -----------------------------
def predict_folder(
    ckpt_path: str,
    test_dir: str,
    out_dir: str,
    thr: float = 0.5,
    size: Tuple[int, int] = (512, 512),
    device: Optional[str] = None,
    save_prob: bool = False,
    save_overlay: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if save_prob:
        os.makedirs(os.path.join(out_dir, "prob"), exist_ok=True)
    if save_overlay:
        os.makedirs(os.path.join(out_dir, "overlay"), exist_ok=True)

    model, mean, std, cfg, device_used = load_checkpoint_model(ckpt_path, device=device)

    # If you want to automatically use training size from cfg when available:
    # cfg_size = cfg.get("img_size", None)
    # if cfg_size is not None and isinstance(cfg_size, (list, tuple)) and len(cfg_size) == 2:
    #     size = (int(cfg_size[0]), int(cfg_size[1]))

    img_paths = list_images(test_dir)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {test_dir}")

    print(f"Device: {device_used}")
    print(f"Found {len(img_paths)} test images.")

    for p in img_paths:
        rgb = read_rgb(p)
        prob, mask = predict_mask_single(
            model=model,
            rgb_uint8=rgb,
            mean=mean,
            std=std,
            size=size,
            device=device_used,
            thr=thr,
        )

        stem = os.path.splitext(os.path.basename(p))[0]

        # Save binary mask
        mask_path = os.path.join(out_dir, f"{stem}_mask.png")
        cv2.imwrite(mask_path, mask)  # grayscale 0/255

        # Save probability map (as 8-bit)
        if save_prob:
            prob8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
            prob_path = os.path.join(out_dir, "prob", f"{stem}_prob.png")
            cv2.imwrite(prob_path, prob8)

        # Save overlay
        if save_overlay:
            ov = overlay_mask(rgb, mask, alpha=0.45)
            ov_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
            ov_path = os.path.join(out_dir, "overlay", f"{stem}_overlay.png")
            cv2.imwrite(ov_path, ov_bgr)

    print(f"Done. Masks saved to: {out_dir}")


# -----------------------------
# 6) CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="unet_lab_eri_best.pt", help="Path to .pt checkpoint")
    ap.add_argument("--test_dir", type=str, required=True, help="Folder with test images")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for binary mask")
    ap.add_argument("--size", type=int, nargs=2, default=[512, 512], help="Resize H W for inference")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--save_prob", action="store_true", help="Save probability maps (8-bit PNG)")
    ap.add_argument("--save_overlay", action="store_true", help="Save RGB overlays")
    args = ap.parse_args()

    predict_folder(
        ckpt_path=args.ckpt,
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        thr=float(args.thr),
        size=(int(args.size[0]), int(args.size[1])),
        device=args.device,
        save_prob=bool(args.save_prob),
        save_overlay=bool(args.save_overlay),
    )


if __name__ == "__main__":
    main()
