# src/compute_class_counts.py
import os
import json
import numpy as np
from PIL import Image
from glob import glob
import logging
from pathlib import Path
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_pixels_per_value_in_label(label_file_path):
    try:
        label_image = Image.open(label_file_path).convert("L")
        label_pixels_np = np.array(label_image).astype(np.int32)
        unique_values, counts = np.unique(label_pixels_np, return_counts=True)
        return dict(zip(unique_values.tolist(), counts.tolist()))
    except FileNotFoundError:
        logging.warning(f"Label file not found: {label_file_path}. Skipping.")
        return {}
    except Exception as e:
        logging.error(f"Error processing label file {label_file_path}: {e}")
        return {}

def compute_global_pixel_counts(processed_labels_directory):
    logging.info(f"Scanning processed label files in: {processed_labels_directory}")
    label_dir_path = Path(processed_labels_directory)
    if not label_dir_path.is_dir():
        logging.error(f"Processed labels directory not found: {processed_labels_directory}")
        return {}

    label_file_paths = list(label_dir_path.glob("*.tiff")) + list(label_dir_path.glob("*.png"))
    logging.info(f"Found {len(label_file_paths)} label files to process.")

    global_pixel_counts = {}
    for file_path in label_file_paths:
        pixel_counts = count_pixels_per_value_in_label(file_path)
        for pixel_value, count in pixel_counts.items():
            int_pixel_value = int(pixel_value)
            global_pixel_counts[int_pixel_value] = global_pixel_counts.get(int_pixel_value, 0) + count

    logging.info(f"Finished counting. Original pixel value counts: {global_pixel_counts}")
    return global_pixel_counts

def main():
    processed_labels_dir = config.PROCESSED_TRAIN_LBL_DIR
    global_counts = compute_global_pixel_counts(processed_labels_dir)

    if not global_counts:
        logging.error("No pixel counts generated. Did preprocessing run? Is the directory correct?")
        return

    global_counts_string_keys = {str(k): v for k, v in global_counts.items()}

    output_json_path = config.GLOBAL_PIXEL_COUNTS_JSON_PATH
    try:
        with open(output_json_path, "w") as f:
            json.dump(global_counts_string_keys, f, indent=4)
        logging.info(f"Saved global counts (original values) to {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to save global counts to {output_json_path}: {e}")

if __name__ == "__main__":
    main()