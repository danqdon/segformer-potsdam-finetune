import os
from pathlib import Path
import torch
import collections.abc

SRC_DIR = Path(__file__).parent.resolve()
BASE_PROJECT_DIR = SRC_DIR.parent
DATA_ROOT_DIR = BASE_PROJECT_DIR / "ISPRS_Potsdam" / "ISPRS-Potsdam"

SPLIT_DIR_NAME = "split_postdam_ir_512"
RAW_TRAIN_IMG_CSV = DATA_ROOT_DIR / SPLIT_DIR_NAME / "train" / "images.csv"
RAW_TRAIN_LBL_CSV = DATA_ROOT_DIR / SPLIT_DIR_NAME / "train" / "labels.csv"
RAW_VAL_IMG_CSV = DATA_ROOT_DIR / SPLIT_DIR_NAME / "validation" / "images.csv"
RAW_VAL_LBL_CSV = DATA_ROOT_DIR / SPLIT_DIR_NAME / "validation" / "labels.csv"

PROCESSED_TRAIN_LBL_DIR = DATA_ROOT_DIR / "processed_train_labels"
PROCESSED_VAL_LBL_DIR = DATA_ROOT_DIR / "processed_val_labels"
PROCESSED_DATA_TRAIN_CSV_PATH = DATA_ROOT_DIR / SPLIT_DIR_NAME / "train" / "image_label_paths.csv"
PROCESSED_DATA_VAL_CSV_PATH = DATA_ROOT_DIR / SPLIT_DIR_NAME / "validation" / "image_label_paths.csv"

INPUT_CHANNEL_INDICES = [0, 1, 2]
assert len(INPUT_CHANNEL_INDICES) == 3, "INPUT_CHANNEL_INDICES must contain exactly 3 indices."

ORIGINAL_PIXEL_TO_CLASS_INDEX = { 29: 0, 76: 1, 150: 2, 179: 3, 226: 4, 255: 5 }
NUM_CLASSES = len(ORIGINAL_PIXEL_TO_CLASS_INDEX)
CLASS_INDEX_TO_NAME = { 0: "Impervious surfaces", 1: "Building", 2: "Low vegetation", 3: "Tree", 4: "Car", 5: "Clutter/background" }
assert len(CLASS_INDEX_TO_NAME) == NUM_CLASSES, "CLASS_INDEX_TO_NAME length mismatch!"
ID2LABEL_CONFIG_FOR_MODEL = CLASS_INDEX_TO_NAME
LABEL2ID_CONFIG_FOR_MODEL = {name: idx for idx, name in ID2LABEL_CONFIG_FOR_MODEL.items()}

PRETRAINED_MODEL_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_SAVE_DIR = BASE_PROJECT_DIR / "models"; MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR_NAME = "segformer_potsdam_finetuned"
BEST_MODEL_STATE_DICT_NAME = "segformer_best.pt"
EPOCH_CHECKPOINT_FILENAME_TEMPLATE = "segformer_checkpoint_epoch_{epoch}.pt"
FINAL_MODEL_PATH = MODEL_SAVE_DIR / FINAL_MODEL_DIR_NAME
BEST_MODEL_STATE_DICT_PATH = MODEL_SAVE_DIR / BEST_MODEL_STATE_DICT_NAME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50; TRAIN_BATCH_SIZE = 16; LEARNING_RATE = 6e-5; WEIGHT_DECAY = 0.01
OPTIMIZER_TYPE = "AdamW"; DATALOADER_NUM_WORKERS = os.cpu_count() // 2
CHECKPOINT_SAVE_INTERVAL_EPOCHS = 10; EARLY_STOPPING_PATIENCE_EPOCHS = 7

EVALUATION_RESULTS_DIR = BASE_PROJECT_DIR / "results"; EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NUM_EVALUATION_SAMPLES_TO_VISUALIZE = 5
METRICS_DIR = BASE_PROJECT_DIR / "metrics"; METRICS_DIR.mkdir(parents=True, exist_ok=True)
BEST_METRICS_CSV_PATH = METRICS_DIR / "best_metrics.csv"
ALL_EPOCH_METRICS_CSV_PATH = METRICS_DIR / "all_epoch_metrics.csv"

GLOBAL_PIXEL_COUNTS_JSON_PATH = SRC_DIR / "global_counts.json"

VISUALIZATION_PALETTE = [ [255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0] ]
assert len(VISUALIZATION_PALETTE) == NUM_CLASSES, f"Palette length ({len(VISUALIZATION_PALETTE)}) must match NUM_CLASSES ({NUM_CLASSES})"