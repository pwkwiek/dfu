import os
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.color import rgb2lab


# -----------------------------
# 1) Paths / pairing
# -----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MSK_EXTS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")  # sometimes masks are png/jpg


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def list_pairs(images_dir: str, labels_dir: str) -> Tuple[List[str], List[str]]:
    """
    Pairs images and masks by filename stem.
    Example: images/abc.jpg with labels/abc.png
    """
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    img_paths = sorted(img_paths)

    mask_map = {}
    for ext in MSK_EXTS:
        for p in glob.glob(os.path.join(labels_dir, f"*{ext}")):
            mask_map[_stem(p)] = p

    images, masks = [], []
    missing = 0
    for ip in img_paths:
        s = _stem(ip)
        mp = mask_map.get(s, None)
        if mp is None:
            missing += 1
            continue
        images.append(ip)
        masks.append(mp)

    if len(images) == 0:
        raise RuntimeError(
            f"No pairs found. Check filenames match between:\n  {images_dir}\n  {labels_dir}"
        )

    if missing > 0:
        print(f"[WARN] {missing} images had no matching label by stem in {labels_dir}")

    print(f"Found {len(images)} image-mask pairs.")
    return images, masks


# -----------------------------
# 2) Lab + ERI (NO RGB fed to model)
# -----------------------------
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


# -----------------------------
# 3) Augmentations
# -----------------------------
def random_flip_rotate(x: np.ndarray, y: np.ndarray):
    if random.random() < 0.5:
        x = np.flip(x, axis=1)
        y = np.flip(y, axis=1)
    if random.random() < 0.5:
        x = np.flip(x, axis=0)
        y = np.flip(y, axis=0)
    k = random.randint(0, 3)
    if k:
        x = np.rot90(x, k, axes=(0, 1))
        y = np.rot90(y, k, axes=(0, 1))
    return x.copy(), y.copy()


def random_affine_cv2(x: np.ndarray, y: np.ndarray,
                      max_rotate_deg=12,
                      max_translate=0.06,
                      min_scale=0.9,
                      max_scale=1.1):
    H, W = y.shape[:2]
    angle = random.uniform(-max_rotate_deg, max_rotate_deg)
    scale = random.uniform(min_scale, max_scale)
    tx = random.uniform(-max_translate, max_translate) * W
    ty = random.uniform(-max_translate, max_translate) * H
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[:, 2] += [tx, ty]
    x_w = cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    y_w = cv2.warpAffine(y, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return x_w, y_w


def random_shadow_on_L(x: np.ndarray, p=0.5):
    if random.random() > p:
        return x
    H, W, _ = x.shape
    L = x[..., 0]

    mask = np.ones((H, W), np.float32)
    cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
    ax, ay = random.randint(W // 6, W // 2), random.randint(H // 6, H // 2)
    angle = random.randint(0, 180)
    shadow_strength = random.uniform(0.55, 0.9)

    tmp = np.zeros((H, W), np.uint8)
    cv2.ellipse(tmp, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
    tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=random.uniform(15, 45))
    tmp = tmp.astype(np.float32) / 255.0

    mask = 1.0 - tmp * (1.0 - shadow_strength)
    L = np.clip(L * mask, 0.0, 1.0)

    out = x.copy()
    out[..., 0] = L
    return out


def random_lab_eri_jitter(x: np.ndarray,
                          p=0.9,
                          L_gamma=(0.85, 1.20),
                          L_shift=(-0.08, 0.08),
                          ab_scale=(0.85, 1.20),
                          ab_shift=(-0.06, 0.06),
                          eri_scale=(0.85, 1.25),
                          eri_shift=(-0.06, 0.06),
                          noise_std=(0.0, 0.03)):
    if random.random() > p:
        return x

    L, a, b, eri = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

    g = random.uniform(*L_gamma)
    s = random.uniform(*L_shift)
    L = np.clip((L ** g) + s, 0.0, 1.0)

    ab_s = random.uniform(*ab_scale)
    a = np.clip(a * ab_s + random.uniform(*ab_shift), -1.0, 1.0)
    b = np.clip(b * ab_s + random.uniform(*ab_shift), -1.0, 1.0)

    e_s = random.uniform(*eri_scale)
    eri = np.clip(eri * e_s + random.uniform(*eri_shift), -1.0, 1.0)

    ns = random.uniform(*noise_std)
    if ns > 0:
        n = np.random.normal(0.0, ns, size=x.shape[:2]).astype(np.float32)
        L = np.clip(L + n, 0.0, 1.0)

    return np.stack([L, a, b, eri], axis=-1).astype(np.float32)


# -----------------------------
# 4) Dataset
# -----------------------------
class DFUDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 mask_paths: List[str],
                 size: Tuple[int, int] = (512, 512),
                 train: bool = True,
                 channel_mean: Optional[np.ndarray] = None,
                 channel_std: Optional[np.ndarray] = None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size
        self.train = train
        self.mean = channel_mean
        self.std = channel_std

    def __len__(self):
        return len(self.image_paths)

    def _read_rgb(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
        # robust binarize
        return (m > 127).astype(np.uint8)

    def __getitem__(self, idx: int):
        rgb = self._read_rgb(self.image_paths[idx])
        y = self._read_mask(self.mask_paths[idx])

        H, W = self.size
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        y = cv2.resize(y, (W, H), interpolation=cv2.INTER_NEAREST)

        x = rgb_uint8_to_lab_eri(rgb)  # HxWx4

        if self.train:
            x, y = random_flip_rotate(x, y)
            if random.random() < 0.7:
                x, y = random_affine_cv2(x, y)
            x = random_shadow_on_L(x, p=0.5)
            x = random_lab_eri_jitter(x, p=0.9)

        if self.mean is not None and self.std is not None:
            x = normalize_per_channel(x, self.mean, self.std)

        x_t = torch.from_numpy(np.transpose(x, (2, 0, 1))).float()  # (4,H,W)
        y_t = torch.from_numpy(y[None, ...]).float()                # (1,H,W)
        return x_t, y_t


def compute_channel_stats(image_paths: List[str], size=(512, 512), max_samples=200):
    paths = image_paths[:]
    random.shuffle(paths)
    paths = paths[:min(max_samples, len(paths))]

    acc = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        x = rgb_uint8_to_lab_eri(rgb)
        acc.append(x.reshape(-1, 4))

    acc = np.concatenate(acc, axis=0).astype(np.float64)
    mean = acc.mean(axis=0).astype(np.float32)
    std = acc.std(axis=0).astype(np.float32) + 1e-6
    return mean, std


# -----------------------------
# 5) U-Net
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
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
# 6) Losses
# -----------------------------
def soft_dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps=1e-6):
    inter = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()

@torch.no_grad()
def iou_score_from_logits(logits: torch.Tensor, y: torch.Tensor, thr=0.5, eps=1e-6) -> float:
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * y).sum(dim=(2, 3))
    union = (p + y - p * y).sum(dim=(2, 3))
    iou = ((inter + eps) / (union + eps)).mean().item()
    return iou

@torch.no_grad()
def pixel_acc_from_logits(logits: torch.Tensor, y: torch.Tensor, thr=0.5) -> float:
    p = (torch.sigmoid(logits) > thr).float()
    correct = (p == y).float().mean().item()
    return correct


def sobel_mag(x: torch.Tensor):
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def grad_consistency_loss(logits: torch.Tensor, x_lab_eri: torch.Tensor, lam=0.05):
    """
    Align predicted mask edges with Lab(a,b) edges.
    logits: (B,1,H,W)
    x_lab_eri: (B,4,H,W) [L,a,b,eri]
    """
    p = torch.sigmoid(logits)

    a = x_lab_eri[:, 1:2]
    b = x_lab_eri[:, 2:3]
    Iab = torch.cat([a, b], dim=1)

    grad_p = sobel_mag(p)
    grad_I = sobel_mag(Iab).detach()

    # boundary band weighting to avoid chasing texture
    band = ((p > 0.2) & (p < 0.8)).float()
    w = 0.5 * band + 0.5 * torch.clamp(grad_p / (grad_p.amax(dim=(2, 3), keepdim=True) + 1e-8), 0, 1)

    return lam * torch.mean(w * torch.abs(grad_p - grad_I))


# -----------------------------
# 7) Metrics
# -----------------------------
@torch.no_grad()
def dice_score_from_logits(logits: torch.Tensor, y: torch.Tensor, thr=0.5, eps=1e-6) -> float:
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * y).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
    dice = ((2 * inter + eps) / (denom + eps)).mean().item()
    return dice


# -----------------------------
# 8) Training
# -----------------------------
@dataclass
class TrainConfig:
    img_size: Tuple[int, int] = (512, 512)
    batch_size: int = 6
    num_workers: int = 2
    lr: float = 2e-4
    epochs: int = 50
    lam_grad: float = 0.05
    base_channels: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, cfg: TrainConfig, epoch: int = 1):
    model.train()
    bce = nn.BCEWithLogitsLoss()

    total_loss, total_dice, total_iou, total_acc, n = 0.0, 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Train {epoch}", leave=True)
    for x, y in pbar:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        logits = model(x)

        loss_seg = bce(logits, y) + soft_dice_loss(torch.sigmoid(logits), y)
        loss_g = grad_consistency_loss(logits, x, lam=cfg.lam_grad)
        loss = loss_seg + loss_g

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        d = dice_score_from_logits(logits, y)
        i = iou_score_from_logits(logits, y)
        a = pixel_acc_from_logits(logits, y)

        total_loss += loss.item() * bs
        total_dice += d * bs
        total_iou  += i * bs
        total_acc  += a * bs
        n += bs

        pbar.set_postfix(loss=loss.item(), dice=d, iou=i)

    return (total_loss / n, total_dice / n, total_iou / n, total_acc / n)


@torch.no_grad()
def validate(model, loader, cfg: TrainConfig, epoch: int = 1):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss, total_dice, total_iou, total_acc, n = 0.0, 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Val {epoch}", leave=True)
    for x, y in pbar:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        logits = model(x)

        loss_seg = bce(logits, y) + soft_dice_loss(torch.sigmoid(logits), y)
        loss_g = grad_consistency_loss(logits, x, lam=cfg.lam_grad)
        loss = loss_seg + loss_g

        bs = x.size(0)
        d = dice_score_from_logits(logits, y)
        i = iou_score_from_logits(logits, y)
        a = pixel_acc_from_logits(logits, y)

        total_loss += loss.item() * bs
        total_dice += d * bs
        total_iou  += i * bs
        total_acc  += a * bs
        n += bs

        pbar.set_postfix(loss=loss.item(), dice=d, iou=i)

    return (total_loss / n, total_dice / n, total_iou / n, total_acc / n)



def run_training(train_images_dir: str, train_labels_dir: str,
                 val_images_dir: str, val_labels_dir: str,
                 cfg: TrainConfig,
                 save_path: str = "unet_lab_eri_best.pt",
                 thr: float = 0.5,
                 show_plots: bool = True):
    """
    Runs training and returns:
      ckpt_path, history_dict

    history_dict keys:
      train_loss, val_loss,
      train_dice, val_dice,
      train_iou,  val_iou,
      train_acc,  val_acc
    """

    # --- local helper metrics (uses your existing dice_score_from_logits) ---
    @torch.no_grad()
    def iou_score_from_logits(logits: torch.Tensor, y: torch.Tensor, thr=0.5, eps=1e-6) -> float:
        p = (torch.sigmoid(logits) > thr).float()
        inter = (p * y).sum(dim=(2, 3))
        union = (p + y - p * y).sum(dim=(2, 3))
        iou = ((inter + eps) / (union + eps)).mean().item()
        return iou

    @torch.no_grad()
    def pixel_acc_from_logits(logits: torch.Tensor, y: torch.Tensor, thr=0.5) -> float:
        p = (torch.sigmoid(logits) > thr).float()
        return (p == y).float().mean().item()

    # --- tqdm + plotting (imports here so your file still runs if you don't use them) ---
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    # --- pair data ---
    train_images, train_masks = list_pairs(train_images_dir, train_labels_dir)
    val_images, val_masks = list_pairs(val_images_dir, val_labels_dir)

    # --- compute normalization stats ---
    mean, std = compute_channel_stats(train_images, size=cfg.img_size, max_samples=200)
    print("Channel mean [L,a,b,eri]:", mean)
    print("Channel std  [L,a,b,eri]:", std)

    # --- datasets/loaders ---
    ds_tr = DFUDataset(train_images, train_masks, size=cfg.img_size, train=True,
                       channel_mean=mean, channel_std=std)
    ds_va = DFUDataset(val_images, val_masks, size=cfg.img_size, train=False,
                       channel_mean=mean, channel_std=std)

    # Windows tip: if you had crashes/hangs, set cfg.num_workers=0
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                       pin_memory=True)

    # --- model/optim ---
    model = UNet(in_channels=4, base=cfg.base_channels).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    bce = nn.BCEWithLogitsLoss()
    best_dice = -1.0

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [],  "val_iou": [],
        "train_acc": [],  "val_acc": [],
    }

    def train_one_epoch_with_bar(epoch: int):
        model.train()
        total_loss = total_dice = total_iou = total_acc = 0.0
        n = 0

        pbar = tqdm(dl_tr, desc=f"Train {epoch}", leave=True)
        for x, y in pbar:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            logits = model(x)
            loss_seg = bce(logits, y) + soft_dice_loss(torch.sigmoid(logits), y)
            loss_g = grad_consistency_loss(logits, x, lam=cfg.lam_grad)
            loss = loss_seg + loss_g

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x.size(0)
            d = dice_score_from_logits(logits, y, thr=thr)
            i = iou_score_from_logits(logits, y, thr=thr)
            a = pixel_acc_from_logits(logits, y, thr=thr)

            total_loss += loss.item() * bs
            total_dice += d * bs
            total_iou  += i * bs
            total_acc  += a * bs
            n += bs

            pbar.set_postfix(loss=float(loss.item()), dice=float(d), iou=float(i))

        return total_loss / n, total_dice / n, total_iou / n, total_acc / n

    @torch.no_grad()
    def validate_with_bar(epoch: int):
        model.eval()
        total_loss = total_dice = total_iou = total_acc = 0.0
        n = 0

        pbar = tqdm(dl_va, desc=f"Val {epoch}", leave=True)
        for x, y in pbar:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            logits = model(x)
            loss_seg = bce(logits, y) + soft_dice_loss(torch.sigmoid(logits), y)
            loss_g = grad_consistency_loss(logits, x, lam=cfg.lam_grad)
            loss = loss_seg + loss_g

            bs = x.size(0)
            d = dice_score_from_logits(logits, y, thr=thr)
            i = iou_score_from_logits(logits, y, thr=thr)
            a = pixel_acc_from_logits(logits, y, thr=thr)

            total_loss += loss.item() * bs
            total_dice += d * bs
            total_iou  += i * bs
            total_acc  += a * bs
            n += bs

            pbar.set_postfix(loss=float(loss.item()), dice=float(d), iou=float(i))

        return total_loss / n, total_dice / n, total_iou / n, total_acc / n

    # --- training loop ---
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_dice, tr_iou, tr_acc = train_one_epoch_with_bar(epoch)
        va_loss, va_dice, va_iou, va_acc = validate_with_bar(epoch)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(va_iou)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss {tr_loss:.4f} dice {tr_dice:.4f} iou {tr_iou:.4f} | "
            f"val loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f}"
        )

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "cfg": cfg.__dict__,
                "best_dice": best_dice,
                "history": history,
            }, save_path)
            print(f"  ✓ saved best: {save_path} (best val dice={best_dice:.4f})")

        # Optional: show plots every epoch (can be slow); by default we plot at end only
        # if show_plots:
        #     plt.figure(); plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="val"); plt.legend(); plt.show()

    print("Done. Best val dice:", best_dice)

    # --- plots at end ---
    if show_plots:
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure()
        plt.plot(epochs, history["train_loss"], label="train loss")
        plt.plot(epochs, history["val_loss"], label="val loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, history["train_dice"], label="train dice")
        plt.plot(epochs, history["val_dice"], label="val dice")
        plt.xlabel("epoch"); plt.ylabel("dice"); plt.title("Dice"); plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, history["train_iou"], label="train IoU")
        plt.plot(epochs, history["val_iou"], label="val IoU")
        plt.xlabel("epoch"); plt.ylabel("IoU"); plt.title("IoU"); plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, history["train_acc"], label="train pixel acc")
        plt.plot(epochs, history["val_acc"], label="val pixel acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Pixel Accuracy"); plt.legend()
        plt.show()

    return save_path, history

