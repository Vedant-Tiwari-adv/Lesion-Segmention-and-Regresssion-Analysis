import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================
class BratsNPZDataset(Dataset):
    def __init__(self, npz_dir, augment=False, modality_dropout_prob=0.1, limit=None):
        self.npz_dir = Path(npz_dir)
        self.files = list(self.npz_dir.glob("*.npz"))
        if limit:
            self.files = self.files[:limit]
        self.augment = augment
        self.modality_dropout_prob = modality_dropout_prob

    def __len__(self):
        return len(self.files)

    def random_flip_3d(self, x):
        for axis in [0, 1, 2]:
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=axis).copy()
        return x

    def random_intensity_jitter(self, volume, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
        scale = np.random.uniform(*scale_range)
        shift = np.random.uniform(*shift_range)
        return volume * scale + shift

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        modalities = ['t1', 't2', 'flair']
        imgs = np.stack([data[m] for m in modalities], axis=0)
        mask = data['mask']

        presence = np.ones(len(modalities), np.float32)

        if self.augment:
            for i in range(len(modalities)):
                if np.random.rand() < self.modality_dropout_prob:
                    imgs[i] = np.zeros_like(imgs[i])
                    presence[i] = 0.0

            imgs = self.random_flip_3d(imgs)
            mask = self.random_flip_3d(mask)

        presence_channels = np.stack([np.ones_like(mask) * p for p in presence], axis=0)
        inp = np.concatenate([imgs, presence_channels], axis=0)

        return {
            'image': torch.from_numpy(inp).float(),
            'mask': torch.from_numpy(mask[None]).float(),
            'name': file_path.stem
        }


# ============================================================
# 3D UNet Model
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, base=16, out_channels=1):
        super().__init__()
        f = base
        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bot = ConvBlock(f * 4, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, 2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, 2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, 2)
        self.dec1 = ConvBlock(f * 2, f)
        self.outc = nn.Conv3d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bot(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.outc(d1)


# ============================================================
# Loss and Metrics
# ============================================================
def dice_loss_logits(pred, target, eps=1e-6):
    prob = torch.sigmoid(pred)
    num = 2 * (prob * target).sum((2, 3, 4))
    den = prob.sum((2, 3, 4)) + target.sum((2, 3, 4)) + eps
    return (1 - num / den).mean()


def compute_metrics(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    tp = (pred_bin * target_bin).sum((2, 3, 4))
    fp = (pred_bin * (1 - target_bin)).sum((2, 3, 4))
    fn = ((1 - pred_bin) * target_bin).sum((2, 3, 4))
    tn = ((1 - pred_bin) * (1 - target_bin)).sum((2, 3, 4))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return dice.mean().item(), iou.mean().item(), precision.mean().item(), recall.mean().item(), accuracy.mean().item()


# ============================================================
# Training + Evaluation Loop (with test metrics printed)
# ============================================================
def train_and_test(train_loader, val_loader, test_loader, model, device, epochs=3, lr=1e-3, save_dir="."):
    save_dir = Path(save_dir)
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)

    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_dice = 0.0

    history = {
        'train_loss': [],
        'val_dice': [], 'val_iou': [], 'val_prec': [], 'val_rec': [], 'val_acc': [],
        'test_dice': [], 'test_iou': [], 'test_prec': [], 'test_rec': [], 'test_acc': []
    }

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {ep}"):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(imgs)
                loss = dice_loss_logits(out, masks) + nn.BCEWithLogitsLoss()(out, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_losses.append(loss.item())

        mean_train_loss = np.mean(train_losses)
        history['train_loss'].append(mean_train_loss)

        # ---------- Validation ----------
        model.eval()
        dices, ious, precs, recs, accs = [], [], [], [], []
        with torch.no_grad():
            for b in val_loader:
                imgs = b['image'].to(device)
                masks = b['mask'].to(device)
                preds = torch.sigmoid(model(imgs))
                d, i, p, r, a = compute_metrics(preds, masks)
                dices.append(d); ious.append(i); precs.append(p); recs.append(r); accs.append(a)

        md, mi, mp, mr, ma = map(np.mean, [dices, ious, precs, recs, accs])
        history['val_dice'].append(md)
        history['val_iou'].append(mi)
        history['val_prec'].append(mp)
        history['val_rec'].append(mr)
        history['val_acc'].append(ma)

        # ---------- Test Evaluation ----------
        dices, ious, precs, recs, accs = [], [], [], [], []
        with torch.no_grad():
            for b in test_loader:
                imgs = b['image'].to(device)
                masks = b['mask'].to(device)
                preds = torch.sigmoid(model(imgs))
                d, i, p, r, a = compute_metrics(preds, masks)
                dices.append(d); ious.append(i); precs.append(p); recs.append(r); accs.append(a)

        td, ti, tp, tr, ta = map(np.mean, [dices, ious, precs, recs, accs])
        history['test_dice'].append(td)
        history['test_iou'].append(ti)
        history['test_prec'].append(tp)
        history['test_rec'].append(tr)
        history['test_acc'].append(ta)

        # ✅ Print all 5 test metrics clearly
        print(f"\nEpoch {ep} Summary:")
        print(f"  Train Loss   : {mean_train_loss:.4f}")
        print(f"  Validation   → Dice={md:.4f}, IoU={mi:.4f}, Precision={mp:.4f}, Recall={mr:.4f}, Accuracy={ma:.4f}")
        print(f"  Test         → Dice={td:.4f}, IoU={ti:.4f}, Precision={tp:.4f}, Recall={tr:.4f}, Accuracy={ta:.4f}\n")

        if md > best_dice:
            best_dice = md
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print("✔ Saved improved model")

    # ---------- Plot Train vs Test ----------
    metrics_to_plot = ["val_dice", "val_iou", "val_prec", "val_rec", "val_acc"]
    for key in metrics_to_plot:
        plt.figure()
        plt.plot(history[key], label=f"Validation {key.split('_')[1]}")
        test_key = key.replace("val_", "test_")
        plt.plot(history[test_key], label=f"Test {key.split('_')[1]}")
        plt.title(f"{key.split('_')[1].upper()} Curve")
        plt.xlabel("Epoch")
        plt.ylabel(key.split('_')[1].capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(results_dir / f"{key.replace('val_', '')}_train_test.png")
        plt.close()

    print(f"✅ Training + Testing complete. Best Validation Dice={best_dice:.4f}")
    print("All plots and model saved in:", results_dir)


# ============================================================
# Main Entry
# ============================================================
if __name__ == "__main__":
    base = Path(r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Reorg_Flair")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    train_ds = BratsNPZDataset(base / "preprocessed_train", augment=True, modality_dropout_prob=0.2, limit=None)
    val_ds = BratsNPZDataset(base / "preprocessed_validation", augment=False, limit=None)
    test_ds = BratsNPZDataset(base / "preprocessed_test", augment=False, limit=None)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    in_ch = 6  # 3 modalities + 3 presence channels
    model = UNet3D(in_ch, base=16, out_channels=1)

    train_and_test(train_dl, val_dl, test_dl, model, device, epochs=100, lr=1e-3, save_dir=base)
