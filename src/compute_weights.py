# src/compute_weights.py
import json
import torch
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_class_weights_from_counts(counts_json_path):
    try:
        with open(counts_json_path, "r") as f:
            global_counts_str_keys = json.load(f)
        logging.info(f"Loaded global counts from {counts_json_path}")
    except FileNotFoundError:
        logging.error(f"Global counts JSON file not found at {counts_json_path}. Run compute_class_counts.py.")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {counts_json_path}.")
        return None
    except Exception as e:
        logging.error(f"Error loading global counts: {e}")
        return None

    label_mapping = config.ORIGINAL_PIXEL_TO_CLASS_INDEX
    num_labels = config.NUM_CLASSES

    counts_per_class_index = {i: 0 for i in range(num_labels)}
    total_pixels_mapped = 0
    unmapped_pixel_counts = {}

    for original_value_str, count in global_counts_str_keys.items():
        try:
            original_value = int(original_value_str)
            if original_value in label_mapping:
                class_index = label_mapping[original_value]
                counts_per_class_index[class_index] += count
                total_pixels_mapped += count
            else:
                unmapped_pixel_counts[original_value] = count
        except ValueError:
            logging.warning(f"Could not convert key '{original_value_str}' to int. Skipping.")
        except Exception as e:
             logging.warning(f"Error processing count for key '{original_value_str}': {e}")

    if unmapped_pixel_counts:
        logging.warning(f"Found counts for original values not in label_mapping: {unmapped_pixel_counts}. These pixels were ignored for weight calculation.")

    if total_pixels_mapped == 0:
        logging.error("Total mapped pixels is zero. Cannot compute weights. Check counts file and label mapping.")
        return None

    logging.info(f"Counts per remapped class index [0..{num_labels-1}]: {counts_per_class_index}")
    logging.info(f"Total pixels considered for weights: {total_pixels_mapped}")

    class_weights = []
    for i in range(num_labels):
        count_for_class_i = counts_per_class_index.get(i, 1)
        if count_for_class_i == 0:
             logging.warning(f"Class index {i} has zero pixels in the counted training data. Assigning default weight component of 1.")
             count_for_class_i = 1

        weight = total_pixels_mapped / (num_labels * count_for_class_i)
        class_weights.append(weight)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    logging.info(f"Computed class weights tensor: {weights_tensor}")
    return weights_tensor

if __name__ == "__main__":
    counts_path = config.GLOBAL_PIXEL_COUNTS_JSON_PATH
    weights = compute_class_weights_from_counts(counts_path)

    if weights is not None:
        logging.info(f"Successfully computed class weights: {weights}")
    else:
        logging.error("Failed to compute class weights.")