# src/train.py
import collections.abc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import numpy as np
import csv  # Import csv module
import logging
import argparse
from transformers import SegformerImageProcessor, get_scheduler
from PIL import Image

import config
from dataset import PotsdamSegmentationDataset, load_processed_data_paths
from model import load_segformer_model, move_model_to_device
from compute_weights import compute_class_weights_from_counts
from utils import map_pixel_values_to_class_indices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_training_arguments():
    parser = argparse.ArgumentParser(description="Train SegFormer model on Potsdam dataset.")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--checkpoint", type=str, default=config.PRETRAINED_MODEL_CHECKPOINT, help="Hugging Face model checkpoint.")
    parser.add_argument("--output-model-name", type=str, default=config.FINAL_MODEL_DIR_NAME, help="Name for the final saved model directory.")
    parser.add_argument("--num-workers", type=int, default=config.DATALOADER_NUM_WORKERS, help="Number of DataLoader workers. Set to 0 for debugging.")
    # New argument to specify channels, as a comma-separated string.
    parser.add_argument("--channels", type=str, default="0,1,2",
                        help="Comma-separated list of channels to use, e.g., '0,1,3'")
    return parser.parse_args()

image_processor = SegformerImageProcessor.from_pretrained(
    config.PRETRAINED_MODEL_CHECKPOINT,
    do_reduce_labels=False
)

def create_input_transform(image, label):
    try:
        image_arr = np.array(image)
        if image_arr.ndim == 3: 
            if any(ch >= image_arr.shape[2] for ch in config.CHANNELS_TO_USE):
                raise ValueError(f"Image has {image_arr.shape[2]} channels, but requested channel indices {config.CHANNELS_TO_USE} require at least {max(config.CHANNELS_TO_USE)+1} channels.")
            selected_arr = image_arr[..., config.CHANNELS_TO_USE]
            image = Image.fromarray(selected_arr)
        else:
            raise ValueError("Input image does not have 3 dimensions as expected.")
        label_np = np.array(label).astype(np.int32)
        label_remapped = map_pixel_values_to_class_indices(label_np, config.ORIGINAL_PIXEL_TO_CLASS_INDEX)
        encoded_inputs = image_processor(image, segmentation_maps=label_remapped, return_tensors="pt")
        if not isinstance(encoded_inputs, collections.abc.Mapping):
            logging.error(f"Processor did not return a dict-like object. Got: {type(encoded_inputs)}")
            return None
        if "pixel_values" not in encoded_inputs or "labels" not in encoded_inputs:
            logging.error(f"Processor output missing required keys 'pixel_values' or 'labels'. Got keys: {list(encoded_inputs.keys())}")
            return None
        if not isinstance(encoded_inputs["pixel_values"], torch.Tensor) or not isinstance(encoded_inputs["labels"], torch.Tensor):
            logging.error(f"Processor output values are not Tensors. Got types: pixel_values={type(encoded_inputs['pixel_values'])}, labels={type(encoded_inputs['labels'])}")
            return None
        return encoded_inputs
    except Exception as e:
        logging.error(f"Error during create_input_transform: {e}", exc_info=True)
        raise e

def run_training(arguments):
    # Override channel selection based on CLI argument.
    try:
        new_channels = list(map(int, arguments.channels.split(",")))
    except Exception as e:
        logging.error(f"Error parsing channels argument: {arguments.channels}. Using default. Exception: {e}")
        new_channels = [0, 1, 2]
    config.CHANNELS_TO_USE = new_channels
    config.NUM_INPUT_CHANNELS = len(new_channels)
    config.CHANNEL_COMBINATION_NAME = "_".join(str(ch) for ch in new_channels)
    # Update paths for saving models based on new channel combination.
    config.FINAL_MODEL_PATH = config.MODEL_SAVE_DIR / config.CHANNEL_COMBINATION_NAME / config.FINAL_MODEL_DIR_NAME
    config.BEST_MODEL_STATE_DICT_PATH = config.MODEL_SAVE_DIR / config.CHANNEL_COMBINATION_NAME / config.BEST_MODEL_STATE_DICT_NAME

    logging.info(f"Using channel combination: {config.CHANNELS_TO_USE}")
    logging.info(f"Models will be saved under subfolder: {config.CHANNEL_COMBINATION_NAME}")

    logging.info("Starting training process...")
    logging.info(f"Using device: {config.DEVICE}")

    try:
        train_df, validation_df = load_processed_data_paths()
    except FileNotFoundError:
        logging.error("Processed data not found. Please run src/preprocess.py first.")
        return
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    train_dataset = PotsdamSegmentationDataset(train_df, transform_function=create_input_transform)
    valid_dataset = PotsdamSegmentationDataset(validation_df, transform_function=create_input_transform)

    pin_memory = config.DEVICE.type == 'cuda'
    dataloader_num_workers = arguments.num_workers
    if dataloader_num_workers == 0:
        logging.warning(f"Running DataLoader with num_workers={dataloader_num_workers} (debugging mode).")
    elif dataloader_num_workers < 0:
        dataloader_num_workers = config.DATALOADER_NUM_WORKERS
        logging.warning(f"Invalid num_workers argument. Using default: {dataloader_num_workers}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=arguments.batch_size, shuffle=True,
        num_workers=dataloader_num_workers, pin_memory=pin_memory
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=arguments.batch_size, shuffle=False,
        num_workers=dataloader_num_workers, pin_memory=pin_memory
    )
    logging.info(f"Dataloaders created with batch size {arguments.batch_size} and {dataloader_num_workers} workers.")

    weights = compute_class_weights_from_counts(config.GLOBAL_PIXEL_COUNTS_JSON_PATH)
    if weights is None:
        logging.warning("Could not compute class weights. Using uniform weights.")
        loss_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        logging.info(f"Using computed class weights: {weights}")
        loss_criterion = nn.CrossEntropyLoss(weight=weights.to(config.DEVICE), ignore_index=-100)

    model = load_segformer_model(model_checkpoint=arguments.checkpoint)
    model = move_model_to_device(model, config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=arguments.lr, weight_decay=config.WEIGHT_DECAY)

    num_training_steps = arguments.epochs * len(train_dataloader) if len(train_dataloader) > 0 else 0
    if num_training_steps == 0:
        logging.warning("Training dataloader is empty, cannot setup LR scheduler properly.")
        learning_rate_scheduler = None
    else:
        learning_rate_scheduler = get_scheduler(
            name="linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps
        )
    logging.info(f"Optimizer: AdamW, LR: {arguments.lr}, Weight Decay: {config.WEIGHT_DECAY}")
    logging.info(f"LR Scheduler: Linear, Total steps: {num_training_steps}")

    all_epoch_metrics_file = config.ALL_EPOCH_METRICS_CSV_PATH
    epoch_log_header = [
        "epoch", "train_loss", "val_loss", "val_miou", "val_macc"
    ] + [f"class_{i}_iou" for i in range(config.NUM_CLASSES)]

    try:
        with open(all_epoch_metrics_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_log_header)
        logging.info(f"Initialized epoch metrics log file: {all_epoch_metrics_file}")
    except Exception as e:
        logging.error(f"Failed to initialize epoch metrics log file {all_epoch_metrics_file}: {e}")

    best_validation_miou = -1.0
    epochs_without_miou_improvement = 0

    # Create checkpoint directory including the channel subfolder.
    checkpoint_dir = config.MODEL_SAVE_DIR / config.CHANNEL_COMBINATION_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(arguments.epochs):
        logging.info(f"\n--- Epoch {epoch+1}/{arguments.epochs} ---")
        model.train()
        total_train_loss = 0.0

        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        try:
            for batch_data in train_progress_bar:
                if batch_data is None:
                    continue

                pixel_values = batch_data["pixel_values"]
                target_labels = batch_data["labels"]

                if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
                    pixel_values = pixel_values.squeeze(1)
                elif pixel_values.ndim != 4:
                    raise ValueError(f"Unexpected training pixel_values dimension: {pixel_values.ndim}")

                if target_labels.ndim == 4 and target_labels.shape[1] == 1:
                    target_labels = target_labels.squeeze(1)
                elif target_labels.ndim != 3:
                    raise ValueError(f"Unexpected training target_labels dimension: {target_labels.ndim}")

                target_labels = target_labels.long()
                pixel_values = pixel_values.to(config.DEVICE)
                target_labels = target_labels.to(config.DEVICE)

                optimizer.zero_grad()
                model_outputs = model(pixel_values=pixel_values, labels=None)
                output_logits = model_outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    output_logits, size=target_labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = loss_criterion(upsampled_logits, target_labels)
                loss.backward()
                optimizer.step()
                if learning_rate_scheduler:
                    learning_rate_scheduler.step()
                total_train_loss += loss.item()
                train_progress_bar.set_postfix(loss=loss.item())
        except Exception as e:
            logging.error(f"Error during training loop iteration: {e}", exc_info=True)
            raise e

        average_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logging.info(f"Average Training Loss: {average_train_loss:.4f}")

        model.eval()
        total_validation_loss = 0.0
        val_metric = evaluate.load("mean_iou")
        validation_progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False)

        try:
            with torch.no_grad():
                for batch_data in validation_progress_bar:
                    if batch_data is None:
                        continue

                    pixel_values = batch_data["pixel_values"]
                    target_labels = batch_data["labels"]

                    if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
                        pixel_values = pixel_values.squeeze(1)
                    elif pixel_values.ndim != 4:
                        raise ValueError(f"Unexpected validation pixel_values dimension: {pixel_values.ndim}")
                    if target_labels.ndim == 4 and target_labels.shape[1] == 1:
                        target_labels = target_labels.squeeze(1)
                    elif target_labels.ndim != 3:
                        raise ValueError(f"Unexpected validation target_labels dimension: {target_labels.ndim}")

                    target_labels = target_labels.long()
                    pixel_values = pixel_values.to(config.DEVICE)
                    target_labels = target_labels.to(config.DEVICE)

                    model_outputs = model(pixel_values=pixel_values, labels=None)
                    output_logits = model_outputs.logits
                    upsampled_logits = nn.functional.interpolate(
                        output_logits, size=target_labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    loss = loss_criterion(upsampled_logits, target_labels)
                    total_validation_loss += loss.item()
                    predicted_labels = upsampled_logits.argmax(dim=1)
                    val_metric.add_batch(predictions=predicted_labels.cpu().numpy(), references=target_labels.cpu().numpy())
                    validation_progress_bar.set_postfix(loss=loss.item())
        except Exception as e:
            logging.error(f"Error during validation loop iteration: {e}", exc_info=True)
            raise e

        average_validation_loss = total_validation_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0

        validation_miou = 0.0
        validation_mean_accuracy = 0.0
        per_category_iou = [0.0] * config.NUM_CLASSES
        try:
            epoch_metrics = val_metric.compute(num_labels=config.NUM_CLASSES, ignore_index=-100, reduce_labels=False)
            validation_miou = epoch_metrics.get("mean_iou", 0)
            validation_mean_accuracy = epoch_metrics.get("mean_accuracy", 0)
            per_category_iou_raw = epoch_metrics.get("per_category_iou", np.array([0.0] * config.NUM_CLASSES))
            per_category_iou = per_category_iou_raw.tolist() if isinstance(per_category_iou_raw, np.ndarray) else per_category_iou_raw
            if not per_category_iou or len(per_category_iou) != config.NUM_CLASSES:
                per_category_iou = [0.0] * config.NUM_CLASSES

            logging.info(f"Validation Loss: {average_validation_loss:.4f} - mIoU: {validation_miou:.4f} - mAcc: {validation_mean_accuracy:.4f}")
            logging.info("Per-class IoU:")
            for i, iou in enumerate(per_category_iou):
                class_name = config.CLASS_INDEX_TO_NAME.get(i, f'Unknown Index {i}')
                logging.info(f"  Class {i} ({class_name}): IoU = {iou:.4f}")

        except Exception as e:
            logging.error(f"Error computing metrics: {e}", exc_info=True)

        try:
            epoch_data_row = [
                epoch + 1, average_train_loss, average_validation_loss,
                validation_miou, validation_mean_accuracy
            ] + per_category_iou
            with open(all_epoch_metrics_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(epoch_data_row)
        except Exception as e:
            logging.error(f"Failed to append epoch {epoch+1} metrics to {all_epoch_metrics_file}: {e}")

        if (epoch + 1) % config.CHECKPOINT_SAVE_INTERVAL_EPOCHS == 0:
            checkpoint_filename = config.EPOCH_CHECKPOINT_FILENAME_TEMPLATE.format(epoch=epoch+1)
            checkpoint_path = checkpoint_dir / checkpoint_filename
            try:
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Saved epoch checkpoint: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Failed to save checkpoint {checkpoint_path}: {e}")

        if validation_miou > best_validation_miou:
            best_validation_miou = validation_miou
            epochs_without_miou_improvement = 0
            try:
                torch.save(model.state_dict(), config.BEST_MODEL_STATE_DICT_PATH)
                logging.info(f"New best mIoU: {validation_miou:.4f}. Saved best model state_dict to {config.BEST_MODEL_STATE_DICT_PATH}")
                try:
                    csv_header = ["epoch", "mIoU", "mean_accuracy"] + [f"class_{i}_iou" for i in range(config.NUM_CLASSES)]
                    iou_list = list(per_category_iou)
                    csv_row = [epoch + 1, validation_miou, validation_mean_accuracy] + iou_list
                    with open(config.BEST_METRICS_CSV_PATH, "w", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(csv_header)
                        csv_writer.writerow(csv_row)
                    logging.info(f"Saved best metrics to {config.BEST_METRICS_CSV_PATH}")
                except Exception as e:
                    logging.error(f"Failed to save best metrics CSV: {e}")
            except Exception as e:
                logging.error(f"Failed to save best model state_dict {config.BEST_MODEL_STATE_DICT_PATH}: {e}")
        else:
            epochs_without_miou_improvement += 1
            logging.info(f"Validation mIoU did not improve for {epochs_without_miou_improvement} epoch(s). Best mIoU: {best_validation_miou:.4f}")
            if epochs_without_miou_improvement >= config.EARLY_STOPPING_PATIENCE_EPOCHS:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs."); break

    logging.info("Training finished.")
    try:
        if config.BEST_MODEL_STATE_DICT_PATH.exists():
            logging.info(f"Loading best model weights from {config.BEST_MODEL_STATE_DICT_PATH} for final save.")
            model.load_state_dict(torch.load(config.BEST_MODEL_STATE_DICT_PATH, map_location=config.DEVICE))
        else:
            logging.warning("Best model checkpoint not found. Saving the model from the last epoch.")
        final_model_save_path = config.FINAL_MODEL_PATH
        final_model_save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_model_save_path)
        image_processor.save_pretrained(final_model_save_path)
        logging.info(f"Saved final model and processor config to {final_model_save_path}")
    except Exception as e:
        logging.error(f"Failed to save final model: {e}")

if __name__ == "__main__":
    cli_args = parse_training_arguments()
    run_training(cli_args)
