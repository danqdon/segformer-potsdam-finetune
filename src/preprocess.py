import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_label_to_grayscale_and_save(input_path, output_path):
    try:
        mask = Image.open(input_path)
        if mask.mode != "L":
            mask = mask.convert("L")
        mask.save(output_path)
        return str(output_path)
    except Exception as e:
        logging.error(f"Failed to process label {input_path}: {e}")
        return None

def run_label_preprocessing_and_create_path_csv(
    image_csv_path, label_csv_path,
    image_col_name, label_col_name,
    output_lbl_dir, output_combined_csv_path
):
    logging.info(f"Processing labels from {label_csv_path}, linking with images from {image_csv_path}")

    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    try:
        images_df = pd.read_csv(image_csv_path)
        labels_df = pd.read_csv(label_csv_path)
        if len(images_df) != len(labels_df):
             logging.error("Image and Label CSV files have different lengths. Cannot proceed.")
             return
    except FileNotFoundError:
        logging.error(f"Error: Input CSV not found at {image_csv_path} or {label_csv_path}")
        return

    original_image_paths = images_df[image_col_name].apply(lambda x: str(config.DATA_ROOT_DIR / x)).tolist()

    labels_df['absolute_path'] = labels_df[label_col_name].apply(lambda x: config.DATA_ROOT_DIR / x)
    processed_label_paths = []
    logging.info(f"Processing {len(labels_df)} labels...")
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing Labels"):
        in_path = row['absolute_path']
        out_path = output_lbl_dir / in_path.name
        processed_path = convert_label_to_grayscale_and_save(in_path, out_path)
        processed_label_paths.append(processed_path if processed_path else None)

    processed_paths_df = pd.DataFrame({
        "image": original_image_paths,
        "label": processed_label_paths
    })

    initial_count = len(processed_paths_df)
    processed_paths_df.dropna(subset=['label'], inplace=True)
    final_count = len(processed_paths_df)
    if initial_count > final_count:
        logging.warning(f"Dropped {initial_count - final_count} samples due to label processing errors.")

    try:
        processed_paths_df.to_csv(output_combined_csv_path, index=False)
        logging.info(f"Saved final data CSV (original images, processed labels) to {output_combined_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save final CSV {output_combined_csv_path}: {e}")

    logging.info("Preprocessing finished.")

if __name__ == "__main__":
    logging.info("--- Running Preprocessing for Training Data ---")
    run_label_preprocessing_and_create_path_csv(
        image_csv_path=config.RAW_TRAIN_IMG_CSV,
        label_csv_path=config.RAW_TRAIN_LBL_CSV,
        image_col_name="images",
        label_col_name="labels",
        output_lbl_dir=config.PROCESSED_TRAIN_LBL_DIR,
        output_combined_csv_path=config.PROCESSED_DATA_TRAIN_CSV_PATH
    )

    logging.info("--- Running Preprocessing for Validation Data ---")
    run_label_preprocessing_and_create_path_csv(
        image_csv_path=config.RAW_VAL_IMG_CSV,
        label_csv_path=config.RAW_VAL_LBL_CSV,
        image_col_name="images",
        label_col_name="labels",
        output_lbl_dir=config.PROCESSED_VAL_LBL_DIR,
        output_combined_csv_path=config.PROCESSED_DATA_VAL_CSV_PATH
    )