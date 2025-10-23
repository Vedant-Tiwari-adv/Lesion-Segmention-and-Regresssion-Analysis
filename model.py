import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Dataset
# -----------------------------
class BratsNPZDataset(Dataset):
    def __init__(self, npz_dir, augment=False, modality_dropout_prob=0.1):
        self.npz_dir = Path(npz_dir)
        self.files = list(self.npz_dir.glob("*.npz"))
        self.augment = augment
        self.modality_dropout_prob = modality_dropout_prob

    def __len__(self):
        return len(self.files)

    def random_flip_3d(self, x):
        axes = [0,1,2]
        for axis in axes:
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=axis).copy()
        return x

    def random_intensity_jitter(self, volume, scale_range=(0.9,1.1), shift_range=(-0.1,0.1)):
        scale = np.random.uniform(*scale_range)
        shift = np.random.uniform(*shift_range)
        return volume*scale + shift

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        modalities = ['t1','t2','flair']
        imgs = np.stack([data[m] for m in modalities], axis=0)
        mask = data['mask']

        presence = np.ones(len(modalities), np.float32)

        # Modality dropout
        if self.augment:
            for i in range(len(modalities)):
                if np.random.rand() < self.modality_dropout_prob:
                    imgs[i] = np.zeros_like(imgs[i])
                    presence[i] = 0.0

        # Random flips
        if self.augment:
            imgs = self.random_flip_3d(imgs)
            mask = self.random_flip_3d(mask)

        # Presence channels
        presence_channels = np.stack([np.ones_like(mask)*p for p in presence], axis=0)
        inp = np.concatenate([imgs, presence_channels], axis=0)

        return {
            'image': torch.from_numpy(inp).float(),
            'mask': torch.from_numpy(mask[None]).float()
        }

# -----------------------------
# 3D U-Net
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, base=16, out_channels=1):
        super().__init__()
        f=base
        self.enc1=ConvBlock(in_channels,f)
        self.pool1=nn.MaxPool3d(2)
        self.enc2=ConvBlock(f,f*2)
        self.pool2=nn.MaxPool3d(2)
        self.enc3=ConvBlock(f*2,f*4)
        self.pool3=nn.MaxPool3d(2)
        self.bot=ConvBlock(f*4,f*8)
        self.up3=nn.ConvTranspose3d(f*8,f*4,2,2)
        self.dec3=ConvBlock(f*8,f*4)
        self.up2=nn.ConvTranspose3d(f*4,f*2,2,2)
        self.dec2=ConvBlock(f*4,f*2)
        self.up1=nn.ConvTranspose3d(f*2,f,2,2)
        self.dec1=ConvBlock(f*2,f)
        self.outc=nn.Conv3d(f,out_channels,1)
    def forward(self,x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool1(e1))
        e3=self.enc3(self.pool2(e2))
        b=self.bot(self.pool3(e3))
        d3=self.dec3(torch.cat([self.up3(b),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.outc(d1)

# -----------------------------
# Loss
# -----------------------------
def dice_loss_logits(pred, target, eps=1e-6):
    prob = torch.sigmoid(pred)
    num = 2*(prob*target).sum((2,3,4))
    den = prob.sum((2,3,4)) + target.sum((2,3,4)) + eps
    return (1 - num/den).mean()

# -----------------------------
# Training loop
# -----------------------------
def train_loop(train_loader, val_loader, model, device, epochs=200, lr=1e-3):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    best_dice=0.0

    for ep in range(1, epochs+1):
        model.train(); losses=[]
        for batch in tqdm(train_loader, desc=f"Epoch {ep}"):
            imgs=batch['image'].to(device)
            masks=batch['mask'].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                out=model(imgs)
                loss=dice_loss_logits(out,masks)+nn.BCEWithLogitsLoss()(out,masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        print(f"Epoch {ep} train loss {np.mean(losses):.4f}")

        # Validation
        model.eval(); dices=[]
        with torch.no_grad():
            for b in val_loader:
                imgs=b['image'].to(device)
                masks=b['mask'].to(device)
                prob=torch.sigmoid(model(imgs))
                num=2*(prob*masks).sum((2,3,4))
                den=prob.sum((2,3,4))+masks.sum((2,3,4))+1e-6
                dices.append(float((num/den).mean()))
        md=float(np.mean(dices))
        print(f"Validation Dice {md:.4f}")
        if md>best_dice:
            best_dice=md
            torch.save(model.state_dict(),"best_model.pth")
            print("Saved improved model.")

    print("Training done. Best Dice:",best_dice)
    print("\nTo use the trained model:\n"
          "model = UNet3D(in_channels, base=16, out_channels=1)\n"
          "model.load_state_dict(torch.load('best_model.pth', map_location=device))\n"
          "model.to(device)\n"
          "model.eval()")

# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    base = Path(r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Reorg_Flair")
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BratsNPZDataset(base / "preprocessed_train", augment=True, modality_dropout_prob=0.2)
    val_ds   = BratsNPZDataset(base / "preprocessed_validation", augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    in_ch = 6  # 3 modalities + 3 presence channels
    model = UNet3D(in_ch, base=16, out_channels=1)

    train_loop(train_dl, val_dl, model, device, epochs=200, lr=1e-3)
