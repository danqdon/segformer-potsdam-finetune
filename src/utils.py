# src/utils.py
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def map_pixel_values_to_class_indices(label_pixel_array, pixel_value_to_index_map):
    try:
        if not isinstance(label_pixel_array, np.ndarray):
            label_pixel_array = np.array(label_pixel_array)

        label_pixel_array_int32 = label_pixel_array.astype(np.int32)
        remapped_array = np.zeros_like(label_pixel_array_int32, dtype=np.int32)

        unique_values_in_array = np.unique(label_pixel_array_int32)
        pixels_remapped_count = 0
        unmapped_pixel_values_found = []

        for original_value in unique_values_in_array:
            if original_value in pixel_value_to_index_map:
                mask = (label_pixel_array_int32 == original_value)
                remapped_array[mask] = pixel_value_to_index_map[original_value]
                pixels_remapped_count += mask.sum()
            else:
                 unmapped_pixel_values_found.append(original_value)
                 pass # Default value is 0 from initialization

        if unmapped_pixel_values_found:
            logging.warning(f"Values {unmapped_pixel_values_found} found in label array but not in mapping {list(pixel_value_to_index_map.keys())}. They were mapped to 0.")

        return remapped_array

    except Exception as e:
        logging.error(f"Error during label remapping: {e}")
        raise