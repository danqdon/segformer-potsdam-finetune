# src/dataset.py
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
from utils import map_pixel_values_to_class_indices
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PotsdamSegmentationDataset(Dataset):
    def __init__(self, dataframe_with_paths, transform_function=None):
        self.dataframe = dataframe_with_paths
        self.transform = transform_function
        logging.info(f"Initialized dataset with {len(self.dataframe)} samples.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data_row = self.dataframe.iloc[index]
        image_path = data_row["image"]
        label_path = data_row["label"]

        try:
            # Open the image without converting to RGB to preserve all channels.
            image = Image.open(image_path)
            label = Image.open(label_path).convert("L")
        except FileNotFoundError as e:
            logging.error(f"Error loading file: {e}. Ensure preprocessing was run.")
            raise FileNotFoundError(f"Cannot find file {image_path} or {label_path}. Did you run preprocess.py?") from e
        except Exception as e:
            logging.error(f"Error opening image/label at index {index} ({image_path}, {label_path}): {e}")
            raise RuntimeError(f"Failed to load sample {index}") from e

        if self.transform:
            processed_data = self.transform(image, label)
            if isinstance(processed_data, collections.abc.Mapping) and \
               "pixel_values" in processed_data and \
               "labels" in processed_data:
                if isinstance(processed_data["pixel_values"], torch.Tensor) and \
                   isinstance(processed_data["labels"], torch.Tensor):
                    return processed_data
                else:
                    logging.error(f"Transform function for index {index} returned correct keys but wrong value types: pixel_values={type(processed_data.get('pixel_values'))}, labels={type(processed_data.get('labels'))}")
                    raise TypeError(f"Transform function failed for index {index}: Incorrect value types.")
            else:
                logging.error(f"Transform function failed for index {index}. Expected dict-like with 'pixel_values' and 'labels', got: {type(processed_data)}")
                raise TypeError(f"Transform function failed for index {index}. Check transform function and processor behavior.")
        else:
            return {"image": image, "label": label}

def load_processed_data_paths():
    train_csv_path = config.PROCESSED_TRAIN_CSV_PATH
    val_csv_path = config.PROCESSED_VAL_CSV_PATH

    if not train_csv_path.exists() or not val_csv_path.exists():
        message = (f"Processed CSV files not found at {train_csv_path} or {val_csv_path}. Please run src/preprocess.py first.")
        logging.error(message)
        raise FileNotFoundError(message)

    try:
        train_dataframe = pd.read_csv(train_csv_path)
        validation_dataframe = pd.read_csv(val_csv_path)
        logging.info(f"Loaded {len(train_dataframe)} training samples from {train_csv_path}")
        logging.info(f"Loaded {len(validation_dataframe)} validation samples from {val_csv_path}")
        return train_dataframe, validation_dataframe
    except Exception as e:
        logging.error(f"Failed to load processed CSV files: {e}")
        raise
