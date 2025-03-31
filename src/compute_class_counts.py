import os
import json
import numpy as np
from PIL import Image
from glob import glob

def count_pixels_in_label(label_path):
    """
    Opens a label image (assumed to be a TIFF), converts it to grayscale ('L'),
    and returns a dictionary with unique pixel values and their counts.
    """
    label = Image.open(label_path).convert("L")
    label_np = np.array(label).astype(np.int32)
    unique, counts = np.unique(label_np, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))

def compute_global_counts(labels_dir):
    """
    Scans all TIFF files in the provided directory, counts pixels per unique value,
    and returns a global count dictionary.
    """
    file_list = glob(os.path.join(labels_dir, "*.tiff"))
    global_counts = {}
    for file in file_list:
        counts = count_pixels_in_label(file)
        for value, cnt in counts.items():
            global_counts[value] = global_counts.get(value, 0) + cnt
    return global_counts

def main():
    # Hardcoded directory containing your processed training label TIFF files
    labels_dir = "/home/dquinteiro/Documentos/VSCodeProjects/SegFormer/ISPRS_Potsdam/ISPRS-Potsdam/processed_train_labels"
    global_counts = compute_global_counts(labels_dir)
    print("Global counts:", global_counts)
    
    # Save the computed counts as a JSON file in the same directory as src.
    # This file will be created in the src folder.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(src_dir, "global_counts.json")
    with open(output_path, "w") as f:
        json.dump(global_counts, f)
    print(f"Saved global counts to {output_path}")

if __name__ == "__main__":
    main()
