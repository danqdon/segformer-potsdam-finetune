import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor
from utils import remap_labels

class PotsdamSegmentationDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): A DataFrame with columns "image" and "label" (absolute paths).
            transform: A function to transform (image, label) pair.
        """
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx): #TODO extend for any combination of bands
        row = self.df.iloc[idx]
        image = Image.open(row["image"]).convert("RGB")
        label = Image.open(row["label"]).convert("L")
        if self.transform:
            processed = self.transform(image, label)
            # We only keep "pixel_values" and "labels" (squeeze out the batch dim added by the processor)
            processed = {k: v.squeeze(0) for k, v in processed.items() if k in ["pixel_values", "labels"]}
            return processed
        else:
            return {"image": image, "label": label}

def process_images_in_dataframe(df, col, output_dir):
    """Convert images to RGB (if they have 4 channels) and save them in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    new_paths = []
    for path in df[col]:
        img = Image.open(path)
        if len(img.getbands()) == 4:
            img = img.convert("RGB")
        new_path = os.path.join(output_dir, os.path.basename(path))
        img.save(new_path)
        new_paths.append(new_path)
    df[col] = new_paths
    return df

def process_labels_in_dataframe(df, col, output_dir):
    """Convert segmentation maps to mode 'L' (1 channel) and save them in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    new_paths = []
    for path in df[col]:
        mask = Image.open(path)
        if mask.mode != "L":
            mask = mask.convert("L")
        new_path = os.path.join(output_dir, os.path.basename(path))
        mask.save(new_path)
        new_paths.append(new_path)
    df[col] = new_paths
    return df

def load_data(base_dir):
    """
    Loads the train and validation CSV files, updates paths, and applies processing.
    Returns two DataFrames: train_df and validation_df.
    """
    base_dir = Path(base_dir)
    train_images_csv = base_dir / "split_postdam_ir_512/train/images.csv"
    train_labels_csv = base_dir / "split_postdam_ir_512/train/labels.csv"
    val_images_csv = base_dir / "split_postdam_ir_512/validation/images.csv"
    val_labels_csv = base_dir / "split_postdam_ir_512/validation/labels.csv"

    train_images_df = pd.read_csv(train_images_csv)
    train_labels_df = pd.read_csv(train_labels_csv)
    val_images_df = pd.read_csv(val_images_csv)
    val_labels_df = pd.read_csv(val_labels_csv)

    train_images_df["image"] = train_images_df["images"].apply(lambda x: str(base_dir / x))
    train_labels_df["label"] = train_labels_df["labels"].apply(lambda x: str(base_dir / x))
    val_images_df["image"] = val_images_df["images"].apply(lambda x: str(base_dir / x))
    val_labels_df["label"] = val_labels_df["labels"].apply(lambda x: str(base_dir / x))

    processed_train_img_dir = str(base_dir / "processed_train_images")
    processed_val_img_dir = str(base_dir / "processed_val_images")
    processed_train_lbl_dir = str(base_dir / "processed_train_labels")
    processed_val_lbl_dir = str(base_dir / "processed_val_labels")

    train_images_df = process_images_in_dataframe(train_images_df, "image", processed_train_img_dir)
    val_images_df = process_images_in_dataframe(val_images_df, "image", processed_val_img_dir)
    train_labels_df = process_labels_in_dataframe(train_labels_df, "label", processed_train_lbl_dir)
    val_labels_df = process_labels_in_dataframe(val_labels_df, "label", processed_val_lbl_dir)

    train_df = pd.DataFrame({
        "image": train_images_df["image"],
        "label": train_labels_df["label"]
    })
    validation_df = pd.DataFrame({
        "image": val_images_df["image"],
        "label": val_labels_df["label"]
    })

    # Optionally, save processed CSVs
    train_df.to_csv(base_dir / "split_postdam_ir_512/train/processed_images_labels.csv", index=False)
    validation_df.to_csv(base_dir / "split_postdam_ir_512/validation/processed_images_labels.csv", index=False)

    return train_df, validation_df
