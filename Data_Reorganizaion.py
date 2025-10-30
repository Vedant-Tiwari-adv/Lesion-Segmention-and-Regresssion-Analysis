import os
import shutil
import glob
import random
from tqdm import tqdm

def organize_brats_data(source_dir, dest_dir, mask_keyword='Flair'):
    """
    Reorganizes the dataset into train/validation/test folders
    using the same dataset (patient-1 to patient-60) for all splits.
    """
    # --- 1. Basic Setup and Validation ---
    print(f"Starting dataset reorganization (SAME-DATA SPLIT MODE)...")
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

    # --- 3. Discover Patients (1–60 only) ---
    patient_folders = [
        d for d in os.listdir(source_dir)
        if (
            os.path.isdir(os.path.join(source_dir, d))
            and d.lower().startswith('patient-')
            and d.split('-')[1].isdigit()
            and 1 <= int(d.split('-')[1]) <= 60
        )
    ]

    if not patient_folders:
        raise ValueError(f"No patient folders found in '{source_dir}'. Please check the path.")

    num_patients = len(patient_folders)
    print(f"Found {num_patients} patients (1–60 range).")

    # --- 4. Assign Same Patients to All Splits ---
    splits = {
        'train': patient_folders,             # full dataset
        'validation': random.sample(patient_folders, max(1, int(0.2 * num_patients))),  # 20% sample
        'test': random.sample(patient_folders, max(1, int(0.2 * num_patients)))         # 20% sample
    }

    print("\nDataset Split (using same data pool):")
    print(f"  - Training:   {len(splits['train'])} patients (full)")
    print(f"  - Validation: {len(splits['validation'])} patients (subset)")
    print(f"  - Test:       {len(splits['test'])} patients (subset)\n")

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
    print("Dataset reorganization complete (same-data mode)!")
    print(f"Organized data is now available in: {dest_dir}")
    print("-----------------------------------------")


if __name__ == '__main__':
    # 🧩 Set your static paths here
    source_dir = r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Dataset_OG"
    dest_dir = r"C:\Personal\Educational\Projects\Lesion-Segmention-and-Regresssion-Analysis\Reorg_Flair"

    mask_keyword = 'Flair'  # Can be 'Flair', 'T1', or 'T2'

    # Run the function
    organize_brats_data(source_dir, dest_dir, mask_keyword)
