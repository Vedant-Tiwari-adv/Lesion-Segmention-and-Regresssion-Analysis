# predict_visualize_mask_fixed.py
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from model import (  # use your training script
    UNet3D, load_nifti_as_numpy, resample_to_spacing,
    clip_and_normalize_zscore, center_crop_or_pad
)

# -----------------------------
# Configuration
# -----------------------------
model_path = r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\best_model.pth"
test_patient_dir = Path(r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Reorg_Flair\validation\Patient-3")
modalities = ['t1', 't2', 'flair']
out_shape = (96, 96, 96)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
in_ch = len(modalities) * 2  # image + presence channels
model = UNet3D(in_ch, base=16, out_channels=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# -----------------------------
# Load MRI volumes
# -----------------------------
imgs, presence = [], []
for mod in modalities:
    path = test_patient_dir / f"{mod}.nii"
    if not path.exists():
        raise FileNotFoundError(f"Missing {mod}.nii in {test_patient_dir}")
    vol, aff = load_nifti_as_numpy(str(path))
    vol = resample_to_spacing(vol, aff)
    vol = clip_and_normalize_zscore(vol)
    vol = center_crop_or_pad(vol, out_shape)
    imgs.append(vol)
    presence.append(1.0)

imgs = np.stack(imgs, axis=0)
presence_channels = np.stack([np.ones(out_shape, np.float32)*p for p in presence], axis=0)
inp = np.concatenate([imgs, presence_channels], axis=0)[None]  # [1, C, D, H, W]

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    inp_tensor = torch.from_numpy(inp).float().to(device)
    pred = torch.sigmoid(model(inp_tensor))[0, 0].cpu().numpy()

mask = (pred > 0.5).astype(np.float32)

# -----------------------------
# Save predicted mask
# -----------------------------
out_path = test_patient_dir / "predicted_mask.nii.gz"
nib.save(nib.Nifti1Image(mask, np.eye(4)), str(out_path))
print(f"🧠 Saved predicted mask to: {out_path}")

# -----------------------------
# Visualization with correct aspect ratio
# -----------------------------
slice_idx = out_shape[0] // 2  # middle slice
h, w = imgs.shape[2], imgs.shape[3]  # height, width

plt.figure(figsize=(12, 12 * h / w))  # maintain aspect ratio

# Original FLAIR slice
plt.subplot(1, 2, 1)
plt.title("FLAIR MRI Slice")
plt.imshow(imgs[2, slice_idx], cmap="gray", origin='lower', aspect='equal')

# Overlay predicted mask
plt.subplot(1, 2, 2)
plt.title("Predicted Mask Overlay")
plt.imshow(imgs[2, slice_idx], cmap="gray", origin='lower', aspect='equal')
plt.imshow(mask[slice_idx], cmap="Reds", alpha=0.4, origin='lower', aspect='equal')

plt.tight_layout()
plt.show()
