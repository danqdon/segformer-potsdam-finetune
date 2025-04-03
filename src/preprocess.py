# src/preprocess.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image_preserve_channels(input_path, output_dir):
    try:
        # Open the original image and save it as-is to preserve all channels.
        img = Image.open(input_path)
        
        # Prepend the name of the parent folder to the original filename
        parent_1 = input_path.parent.name             # 'images'
        parent_2 = input_path.parent.parent.name      # 'postdam_ir_512_2_10'
        prefix = f"{parent_2}_{parent_1}"
        new_filename = f"{prefix}_{input_path.name}"
        
        output_path = output_dir / new_filename
        # Save using a format that supports multiple channels (e.g., TIFF)
        img.save(output_path)
        return str(output_path)
    except Exception as e:
        logging.error(f"Failed to process image {input_path}: {e}")
        return None

def convert_label_to_grayscale_and_save(input_path, output_dir):
    try:
        mask = Image.open(input_path)
        if mask.mode != "L":
            mask = mask.convert("L")
        # Similarly, prefix the parent's name to avoid overwriting.
        prefix = input_path.parent.name
        new_filename = f"{prefix}_{input_path.name}"
        output_path = output_dir / new_filename
        mask.save(output_path)
        return str(output_path)
    except Exception as e:
        logging.error(f"Failed to process label {input_path}: {e}")
        return None

def run_dataset_preprocessing(image_csv_path, label_csv_path, image_col_name, label_col_name,
                              output_img_dir, output_lbl_dir, output_combined_csv_path):
    logging.info(f"Starting preprocessing for files listed in {image_csv_path} and {label_csv_path}")

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    try:
        images_df = pd.read_csv(image_csv_path)
        labels_df = pd.read_csv(label_csv_path)
    except FileNotFoundError:
        logging.error(f"Error: Input CSV not found at {image_csv_path} or {label_csv_path}")
        return

    images_df['absolute_path'] = images_df[image_col_name].apply(lambda x: config.DATA_ROOT_DIR / x)
    labels_df['absolute_path'] = labels_df[label_col_name].apply(lambda x: config.DATA_ROOT_DIR / x)

    processed_image_paths = []
    logging.info(f"Processing {len(images_df)} images...")
    for _, row in tqdm(images_df.iterrows(), total=len(images_df), desc="Processing Images"):
        in_path = row['absolute_path']
        # Use the parent's folder name to generate a unique output filename
        processed_path = process_image_preserve_channels(in_path, output_img_dir)
        processed_image_paths.append(processed_path if processed_path else None)

    processed_label_paths = []
    logging.info(f"Processing {len(labels_df)} labels...")
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing Labels"):
        in_path = row['absolute_path']
        processed_path = convert_label_to_grayscale_and_save(in_path, output_lbl_dir)
        processed_label_paths.append(processed_path if processed_path else None)

    processed_paths_df = pd.DataFrame({
        "image": processed_image_paths,
        "label": processed_label_paths
    })

    initial_count = len(processed_paths_df)
    processed_paths_df.dropna(inplace=True)
    final_count = len(processed_paths_df)
    if initial_count > final_count:
        logging.warning(f"Dropped {initial_count - final_count} samples due to processing errors.")

    try:
        processed_paths_df.to_csv(output_combined_csv_path, index=False)
        logging.info(f"Saved processed data CSV to {output_combined_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save processed CSV {output_combined_csv_path}: {e}")

    logging.info("Preprocessing finished.")

if __name__ == "__main__":
    logging.info("--- Running Preprocessing for Training Data ---")
    run_dataset_preprocessing(
        image_csv_path=config.RAW_TRAIN_IMG_CSV,
        label_csv_path=config.RAW_TRAIN_LBL_CSV,
        image_col_name="images",
        label_col_name="labels",
        output_img_dir=config.PROCESSED_TRAIN_IMG_DIR,
        output_lbl_dir=config.PROCESSED_TRAIN_LBL_DIR,
        output_combined_csv_path=config.PROCESSED_TRAIN_CSV_PATH
    )

    logging.info("--- Running Preprocessing for Validation Data ---")
    run_dataset_preprocessing(
        image_csv_path=config.RAW_VAL_IMG_CSV,
        label_csv_path=config.RAW_VAL_LBL_CSV,
        image_col_name="images",
        label_col_name="labels",
        output_img_dir=config.PROCESSED_VAL_IMG_DIR,
        output_lbl_dir=config.PROCESSED_VAL_LBL_DIR,
        output_combined_csv_path=config.PROCESSED_VAL_CSV_PATH
    )
