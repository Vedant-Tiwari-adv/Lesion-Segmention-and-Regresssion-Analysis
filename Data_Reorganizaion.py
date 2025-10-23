import os
import shutil
import glob
import random
from tqdm import tqdm

def organize_brats_data(source_dir, dest_dir, split_ratio=(0.7, 0.15, 0.15), mask_keyword='Flair'):
    """
    Reorganizes the dataset from a patient-centric structure to a
    train/validation/test split structure suitable for a 3D U-Net pipeline.
    """
    # --- 1. Basic Setup and Validation ---
    print(f"Starting dataset reorganization...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")

    if os.path.exists(dest_dir):
        print(f"Warning: Destination directory '{dest_dir}' already exists. It will be overwritten.")
        shutil.rmtree(dest_dir)

    # --- 2. Create Destination Folders ---
    os.makedirs(dest_dir, exist_ok=True)
    split_folders = ['train', 'validation', 'test']
    for folder in split_folders:
        os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)
    print("Created destination directories.")

    # --- 3. Discover and Shuffle Patients ---
    patient_folders = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and d.lower().startswith('patient-')
    ]
    if not patient_folders:
        raise ValueError(f"No patient folders found in '{source_dir}'. Please check the path.")

    random.shuffle(patient_folders)
    num_patients = len(patient_folders)
    print(f"Found and shuffled {num_patients} patients.")

    # --- 4. Calculate Split Indices ---
    train_end = int(num_patients * split_ratio[0])
    val_end = train_end + int(num_patients * split_ratio[1])
    
    splits = {
        'train': patient_folders[:train_end],
        'validation': patient_folders[train_end:val_end],
        'test': patient_folders[val_end:]
    }

    print("\nDataset Split:")
    print(f"  - Training:   {len(splits['train'])} patients")
    print(f"  - Validation: {len(splits['validation'])} patients")
    print(f"  - Test:       {len(splits['test'])} patients\n")

    # --- 5. Process and Copy Files ---
    for split_name, patient_list in splits.items():
        print(f"--- Processing {split_name.upper()} set ---")
        if not patient_list:
            print(f"No patients allocated for the {split_name} set. Skipping.")
            continue
            
        for patient_name in tqdm(patient_list, desc=f"Copying {split_name} data"):
            patient_id = patient_name.split('-')[-1]
            source_patient_dir = os.path.join(source_dir, patient_name)
            dest_patient_dir = os.path.join(dest_dir, split_name, patient_name)
            os.makedirs(dest_patient_dir, exist_ok=True)

            # Define file patterns to find the necessary scans and mask
            file_patterns = {
                't1': f'*{patient_id}-T1.nii',
                't2': f'*{patient_id}-T2.nii',
                'flair': f'*{patient_id}-Flair.nii',
                'mask': f'*{patient_id}-LesionSeg-{mask_keyword}.nii'
            }

            # Find and copy each required file
            for key, pattern in file_patterns.items():
                search_path = os.path.join(source_patient_dir, pattern)
                files_found = glob.glob(search_path)
                if not files_found:
                    print(f"\nWarning: Could not find file for pattern '{pattern}' in '{source_patient_dir}'. Skipping file.")
                    continue

                source_file_path = files_found[0]
                dest_filename = f"{key}.nii"
                dest_file_path = os.path.join(dest_patient_dir, dest_filename)
                
                shutil.copy(source_file_path, dest_file_path)

    print("\n-----------------------------------------")
    print("Dataset reorganization complete!")
    print(f"Organized data is now available in: {dest_dir}")
    print("-----------------------------------------")


if __name__ == '__main__':
    # 🧩 Set your static paths here
    source_dir = r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Dataset_OG"
    dest_dir = r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Reorg_Flair"

    split_ratio = (0.7, 0.15, 0.15)  # Train / Validation / Test
    mask_keyword = 'Flair'  # Can be 'Flair', 'T1', or 'T2'

    # Run the function
    organize_brats_data(source_dir, dest_dir, split_ratio, mask_keyword)
