import os
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
# Using joint preprocessing to guarantee identical crop coordinates for image and mask
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from data.task_configs import task1_config  # You can also define an isles24_config
from utils.utils import parallel_process
import traceback


def process_isles24_subject(task_info):
    """
    Process a single ISLES2024 subject (ADC + DWI + Lesion Mask).
    """
    folder_name, source_path, pp_config, target_preprocessed, prefix = task_info

    try:
        # Navigate to the Session 2 folder inside derivatives
        ses2_path = join(source_path, "derivatives", folder_name, "ses-02")

        if not os.path.isdir(ses2_path):
            return f"Skipping {folder_name}: No ses-02 directory found."

        # Map ISLES2024 BIDS filenames
        adc_file = join(ses2_path, f"{folder_name}_ses-02_space-ncct_adc.nii.gz")
        dwi_file = join(ses2_path, f"{folder_name}_ses-02_space-ncct_dwi.nii.gz")
        mask_file = join(ses2_path, f"{folder_name}_ses-02_space-ncct_lesion-msk.nii.gz")

        if not all(os.path.exists(f) for f in [adc_file, dwi_file, mask_file]):
            return f"Error: Missing ADC, DWI, or Mask for {folder_name}"

        # Standardize subject ID
        subject_id = folder_name.replace("-", "_")

        # ==========================================
        # UPDATED: Helper function to orient RAS and sanitize
        # ==========================================
        def load_sanitize_and_orient(filepath, is_mask=False):
            # 1. Load the raw NIfTI file
            img = nib.load(filepath)
            
            # 2. Force to closest canonical orientation (RAS+)
            img_ras = nib.as_closest_canonical(img)
            
            # 3. Extract the oriented data array
            data = img_ras.get_fdata()
            
            # 4. Sanitize NaNs and Infs
            if not is_mask:
                # For ADC/DWI: Replace bad math with 0.0
                clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # For masks: Ensure it remains clean integer values (0=background, 1=lesion)
                clean_data = np.nan_to_num(data, nan=0).astype(np.uint8)
            
            # Repackage back into a Nibabel object so Yucca can read the new affine
            return nib.Nifti1Image(clean_data, img_ras.affine, img_ras.header)

        # Load, Orient to RAS, and Stack input modalities
        images = [
            load_sanitize_and_orient(adc_file, is_mask=False), 
            load_sanitize_and_orient(dwi_file, is_mask=False)
        ]
        label = load_sanitize_and_orient(mask_file, is_mask=True)

        # Optional: Keep the safety check just in case there are truly blank images
        for idx, img in enumerate(images):
            if np.std(img.get_fdata()) == 0:
                mod_name = "ADC" if idx == 0 else "DWI"
                return f"Skipped {folder_name}: {mod_name} scan has 0 variance."
        # ==========================================

        # Apply joint preprocessing
        preprocessed_images, preprocessed_label, _ = preprocess_case_for_training_with_label(
            images=images,
            label=label,
            normalization_operation=[pp_config["norm_op"] for _ in range(len(images))],
            allow_missing_modalities=False,
            crop_to_nonzero=pp_config.get("crop_to_nonzero", True),
        )

        # Save preprocessed arrays matching the FOMO naming convention
        base_save_path = join(target_preprocessed, f"{prefix}_{subject_id}")
        np.save(base_save_path + ".npy", preprocessed_images)
        np.save(base_save_path + "_seg.npy", preprocessed_label)

        return f"Processed {folder_name}"

    except Exception as e:
        # repr(e) forces it to print the Exception Type (e.g., AssertionError, ValueError)
        # traceback.format_exc() grabs the full line-by-line crash report
        error_details = traceback.format_exc()
        return f"Error processing {folder_name}: {repr(e)}\n{error_details}"


def convert_and_preprocess_isles24(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess all ISLES2024 subjects in parallel.
    """
    # Using task1_config as the base configuration, matching fomo1.py
    pp_config = task1_config
    task_name = "isles24"  # Or pp_config["task_name"] if dynamically configured
    prefix = "ISLES24"

    derivatives_dir = join(source_path, "derivatives")
    target_preprocessed = join(output_path, task_name)
    ensure_dir_exists(target_preprocessed)

    # Collect only valid subject folders that contain a ses-02 subdirectory
    folder_names = [
        f for f in os.listdir(derivatives_dir)
        if f.startswith("sub-stroke") and os.path.isdir(join(derivatives_dir, f, "ses-02"))
    ]

    assert len(folder_names) > 0, f"No valid ses-02 subjects collected in {derivatives_dir}."

    tasks = [
        (folder_name, source_path, pp_config, target_preprocessed, prefix)
        for folder_name in folder_names
    ]

    parallel_process(
        process_isles24_subject,
        tasks,
        num_workers,
        desc="Preprocessing ISLES24 (ADC + DWI + Mask)",
    )

    print(f"ISLES2024 preprocessing completed. Data saved to {target_preprocessed}")