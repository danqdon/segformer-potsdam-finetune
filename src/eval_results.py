# src/eval_results.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import argparse
import logging
from pathlib import Path
import collections.abc

import config
from model import move_model_to_device # No need for load_model here typically
from utils import map_pixel_values_to_class_indices


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_evaluation_arguments():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer model and visualize results.")
    parser.add_argument(
        "--model-path", type=str, default=str(config.FINAL_MODEL_PATH),
        help="Path to the trained model directory (Hugging Face format)."
    )
    parser.add_argument(
        "--num-samples", type=int, default=config.NUM_EVALUATION_SAMPLES_TO_VISUALIZE,
        help="Number of validation samples to visualize."
    )
    parser.add_argument(
        "--val-csv", type=str, default=str(config.PROCESSED_VAL_CSV_PATH),
        help="Path to the processed validation CSV file."
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(config.EVALUATION_RESULTS_DIR),
        help="Directory to save evaluation figures."
    )
    parser.add_argument(
        "--debug-prints", action='store_true',
        help="Enable detailed printing of unique values."
    )
    return parser.parse_args()


def create_segmentation_overlay(image, segmentation_map, color_palette):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if not isinstance(segmentation_map, np.ndarray):
         segmentation_map = np.array(segmentation_map)

    map_height, map_width = segmentation_map.shape
    color_segmentation = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    for class_index, color in enumerate(color_palette):
        if class_index < len(color_palette):
             color_segmentation[segmentation_map == class_index] = color
        else:
             logging.warning(f"Label index {class_index} out of bounds for palette size {len(color_palette)}.")

    image_np = np.array(image.convert("RGB")).astype(np.float32)
    color_segmentation_np = color_segmentation.astype(np.float32)

    overlay = (image_np * 0.6 + color_segmentation_np * 0.4).astype(np.uint8)
    return Image.fromarray(overlay)

def compute_class_pixel_distribution(segmentation_map, number_of_classes):
    total_pixels = segmentation_map.size
    if total_pixels == 0:
        return {cls_idx: 0.0 for cls_idx in range(number_of_classes)}

    class_percentages = {}
    unique_labels, counts = np.unique(segmentation_map, return_counts=True)
    counts_dict = dict(zip(unique_labels, counts))

    for cls_idx in range(number_of_classes):
        count = counts_dict.get(cls_idx, 0)
        percentages = (100.0 * count / total_pixels) if total_pixels > 0 else 0.0
        class_percentages[cls_idx] = percentages
    return class_percentages


def log_class_distribution(distribution_dict, title_prefix, class_index_to_name_map):
    logging.info(f"{title_prefix} (class %):")
    if not class_index_to_name_map:
        class_index_to_name_map = {}
        logging.warning("Class index to name map not provided for logging distribution.")

    max_len = max(len(name) for name in class_index_to_name_map.values()) if class_index_to_name_map else 10

    for class_index, percentage in distribution_dict.items():
        class_name = class_index_to_name_map.get(class_index, f"Unknown Index {class_index}")
        logging.info(f"  {class_name:<{max_len}} (Index {class_index}): {percentage:>6.2f}%")
    print("")


def run_evaluation_and_visualization(arguments):
    logging.info("Starting evaluation process...")
    logging.info(f"Using device: {config.DEVICE}")
    logging.info(f"Loading model from: {arguments.model_path}")
    logging.info(f"Visualizing {arguments.num_samples} samples.")

    try:
        model = SegformerForSemanticSegmentation.from_pretrained(arguments.model_path)
        image_processor = SegformerImageProcessor.from_pretrained(arguments.model_path)
        model = move_model_to_device(model, config.DEVICE)
        model.eval()
        logging.info("Model and processor loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model or processor from {arguments.model_path}: {e}")
        return

    val_csv_path = Path(arguments.val_csv)
    if not val_csv_path.exists():
        logging.error(f"Validation CSV not found: {val_csv_path}. Run preprocess.py.")
        return
    try:
        validation_dataframe = pd.read_csv(val_csv_path)
        logging.info(f"Loaded {len(validation_dataframe)} validation samples from {val_csv_path}")
    except Exception as e:
        logging.error(f"Failed to load validation CSV {val_csv_path}: {e}")
        return

    num_samples_to_process = arguments.num_samples
    if num_samples_to_process > len(validation_dataframe):
         logging.warning(f"Requested {num_samples_to_process} samples, but only {len(validation_dataframe)} available. Evaluating all.")
         num_samples_to_process = len(validation_dataframe)
    elif num_samples_to_process <= 0:
         logging.error("Number of samples must be positive.")
         return


    results_output_dir = Path(arguments.results_dir)
    results_output_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = config.ORIGINAL_PIXEL_TO_CLASS_INDEX
    num_classes = config.NUM_CLASSES
    color_palette = config.VISUALIZATION_PALETTE
    class_names_map = config.CLASS_INDEX_TO_NAME

    for i in range(num_samples_to_process):
        logging.info(f"\n--- Processing Sample {i+1}/{num_samples_to_process} ---")
        image_file_path = None
        try:
            sample_data_row = validation_dataframe.iloc[i]
            image_file_path = Path(sample_data_row["image"])
            label_file_path = Path(sample_data_row["label"])

            if not image_file_path.exists() or not label_file_path.exists():
                logging.warning(f"Skipping sample {i} due to missing file: {image_file_path} or {label_file_path}")
                continue

            input_image = Image.open(image_file_path).convert("RGB")
            ground_truth_label_image = Image.open(label_file_path).convert("L")
            ground_truth_pixels_original = np.array(ground_truth_label_image).astype(np.int32)

            ground_truth_indices = map_pixel_values_to_class_indices(ground_truth_pixels_original, label_mapping)

            if arguments.debug_prints:
                 logging.info(f"Original unique values in GT label: {np.unique(ground_truth_pixels_original)}")
                 logging.info(f"Remapped unique values in GT label: {np.unique(ground_truth_indices)}")


            model_inputs = image_processor(images=input_image, return_tensors="pt")
            model_inputs = {k: v.to(config.DEVICE) for k, v in model_inputs.items()}

            with torch.no_grad():
                outputs = model(**model_inputs)
                output_logits = outputs.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    output_logits,
                    size=input_image.size[::-1],
                    mode="bilinear",
                    align_corners=False
                )
                predicted_class_indices = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

            if arguments.debug_prints:
                logging.info(f"Unique values in prediction: {np.unique(predicted_class_indices)}")

            ground_truth_distribution = compute_class_pixel_distribution(ground_truth_indices, num_classes)
            prediction_distribution = compute_class_pixel_distribution(predicted_class_indices, num_classes)

            logging.info(f"\nImage {i}: {image_file_path.name}")
            log_class_distribution(ground_truth_distribution, "Ground Truth", class_names_map)
            log_class_distribution(prediction_distribution, "Prediction", class_names_map)

            prediction_overlay = create_segmentation_overlay(input_image, predicted_class_indices, color_palette)
            ground_truth_overlay = create_segmentation_overlay(input_image, ground_truth_indices, color_palette)


            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(input_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            axes[1].imshow(ground_truth_overlay)
            axes[1].set_title("Ground Truth Segmentation")
            axes[1].axis("off")
            axes[2].imshow(prediction_overlay)
            axes[2].set_title("Predicted Segmentation")
            axes[2].axis("off")
            plt.tight_layout()

            output_figure_filename = f"evaluation_sample_{i}_{image_file_path.stem}.png"
            output_figure_path = results_output_dir / output_figure_filename
            try:
                plt.savefig(output_figure_path, bbox_inches="tight")
                logging.info(f"Saved evaluation figure to {output_figure_path}")
            except Exception as e:
                logging.error(f"Failed to save figure {output_figure_path}: {e}")
            plt.close(fig)

        except FileNotFoundError as e:
             logging.error(f"File not found during evaluation of sample {i}: {e}")
        except Exception as e:
            image_name = image_file_path.name if image_file_path else "Unknown"
            logging.error(f"An error occurred processing sample {i} ({image_name}): {e}", exc_info=True)

    logging.info("Evaluation finished.")


if __name__ == "__main__":
    cli_args = parse_evaluation_arguments()
    run_evaluation_and_visualization(cli_args)