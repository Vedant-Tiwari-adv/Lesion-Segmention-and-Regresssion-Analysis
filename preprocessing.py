import os
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------
def load_nifti_as_numpy(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    return data, affine

def resample_to_spacing(volume, affine, target_spacing=(1.0,1.0,1.0)):
    zooms = np.abs(np.diag(affine))[:3]
    zoom_factors = zooms / np.array(target_spacing)
    resampled = zoom(volume, zoom_factors, order=1)
    return resampled

def n4_bias_correction(volume):
    sitk_image = sitk.GetImageFromArray(volume)
    corrected = sitk.N4BiasFieldCorrection(sitk_image)
    return sitk.GetArrayFromImage(corrected)

def clip_and_normalize_zscore(volume, low=1, high=99):
    vmin, vmax = np.percentile(volume, [low, high])
    volume = np.clip(volume, vmin, vmax)
    mean, std = volume.mean(), volume.std()
    if std < 1e-6: std = 1.0
    norm = (volume - mean)/std
    return norm.astype(np.float32)

def center_crop_or_pad(volume, target_shape):
    D,H,W = volume.shape
    d,h,w = target_shape
    pad_before = [max((d-D)//2,0), max((h-H)//2,0), max((w-W)//2,0)]
    pad_after = [max(d-D-pad_before[0],0), max(h-H-pad_before[1],0), max(w-W-pad_before[2],0)]
    volume = np.pad(volume, ((pad_before[0], pad_after[0]),
                             (pad_before[1], pad_after[1]),
                             (pad_before[2], pad_after[2])), mode='constant')
    D,H,W = volume.shape
    sd,sh,sw = (D-d)//2, (H-h)//2, (W-w)//2
    return volume[sd:sd+d, sh:sh+h, sw:sw+w]

# -----------------------------
# Preprocess a single patient
# -----------------------------
def preprocess_patient(patient_dir, modalities=['t1','t2','flair'], use_n4=True, target_shape=(128,128,128)):
    processed = {}
    for mod in modalities:
        path = Path(patient_dir) / f"{mod}.nii"
        if not path.exists():
            processed[mod] = np.zeros(target_shape, np.float32)
            continue
        vol, aff = load_nifti_as_numpy(str(path))
        vol = resample_to_spacing(vol, aff)
        if use_n4:
            vol = n4_bias_correction(vol)
        vol = clip_and_normalize_zscore(vol)
        vol = center_crop_or_pad(vol, target_shape)
        processed[mod] = vol.astype(np.float32)
    
    # mask
    mask_path = Path(patient_dir) / "mask.nii"
    if mask_path.exists():
        mask, aff = load_nifti_as_numpy(str(mask_path))
        mask = resample_to_spacing(mask, aff)
        mask = center_crop_or_pad(mask, target_shape)
        mask = (mask > 0.5).astype(np.float32)
    else:
        mask = np.zeros(target_shape, np.float32)
    
    processed['mask'] = mask
    return processed

# -----------------------------
# Preprocess all patients in folders
# -----------------------------
if __name__=="__main__":
    base_dir = Path(r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Reorg_Flair")
    target_shape = (128,128,128)
    use_n4 = False  # can enable if needed
    
    for split in ["train", "validation", "test"]:
        split_dir = base_dir / split
        output_dir = base_dir / f"preprocessed_{split}"
        output_dir.mkdir(exist_ok=True)
        
        patient_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        for patient_dir in tqdm(patient_dirs, desc=f"Preprocessing {split} patients"):
            data = preprocess_patient(patient_dir, target_shape=target_shape, use_n4=use_n4)
            save_path = output_dir / f"{patient_dir.name}.npz"
            np.savez_compressed(save_path, **data)
