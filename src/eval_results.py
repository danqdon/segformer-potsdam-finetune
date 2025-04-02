import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from model import load_model, move_model
from utils import remap_labels

def visualize_segmentation(image, seg_map, palette):
    h, w = seg_map.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg_map == label] = color
    overlay = (np.array(image).astype(np.float32) * 0.5 + color_seg.astype(np.float32) * 0.5).astype(np.uint8)
    return overlay

def compute_class_distribution(seg_map, num_classes):
    total_pixels = seg_map.size
    percentages = {}
    for cls in range(num_classes):
        count = np.sum(seg_map == cls)
        percentages[cls] = 100 * count / total_pixels
    return percentages

def print_class_distribution(distribution, title):
    print(f"{title} (class %):")
    for cls, perc in distribution.items():
        print(f"  Class {cls}: {perc:.2f}%")
    print("")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    label_mapping = {29: 0, 76: 1, 150: 2, 179: 3, 226: 4, 255: 5}
    num_labels = 6
    original_values = sorted(label_mapping.keys())
    id2label = {i: str(val) for i, val in enumerate(original_values)}
    label2id = {v: k for k, v in id2label.items()}

    model_path = "segformer_potsdam_finetuned"
    model = load_model(num_labels, id2label, label2id)
    model = move_model(model, device)
    model.eval()

    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        do_reduce_labels=False
    )

    palette = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
    ]
    
    base_dir = "/home/dquinteiro/Documentos/VSCodeProjects/SegFormer/ISPRS_Potsdam/ISPRS-Potsdam"
    val_csv = os.path.join(base_dir, "split_postdam_ir_512/validation/processed_images_labels.csv")
    val_df = pd.read_csv(val_csv)

    results_dir = "/home/dquinteiro/Documentos/VSCodeProjects/SegFormer/results"
    os.makedirs(results_dir, exist_ok=True)

    for i in range(5):
        sample_row = val_df.iloc[i]
        test_image_path = sample_row["image"]
        test_label_path = sample_row["label"]
        
        image = Image.open(test_image_path).convert("RGB")
        label = Image.open(test_label_path).convert("L")
        label_np = np.array(label).astype(np.int32)
        print("Valores únicos originales:", np.unique(label_np))
        gt_remapped = remap_labels(label_np, label_mapping)
        print("Valores únicos remapeados:", np.unique(gt_remapped))

        
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        print("Valores únicos en la predicción:", np.unique(predicted))

        # Print class distributions
        gt_dist = compute_class_distribution(gt_remapped, num_labels)
        pred_dist = compute_class_distribution(predicted, num_labels)
        print(f"\nImage {i}: {os.path.basename(test_image_path)}")
        print_class_distribution(gt_dist, "Ground Truth")
        print_class_distribution(pred_dist, "Prediction")

        # Visualizations
        pred_overlay = visualize_segmentation(image, predicted, palette)
        gt_overlay = visualize_segmentation(image, gt_remapped, palette)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth Segmentation")
        axes[1].axis("off")
        
        axes[2].imshow(pred_overlay)
        axes[2].set_title("Predicted Segmentation")
        axes[2].axis("off")
        
        output_path = os.path.join(results_dir, f"evaluation_sample_{i}.png")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved evaluation figure to {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()
