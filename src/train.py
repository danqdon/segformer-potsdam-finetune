import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import numpy as np
import csv

from dataset import PotsdamSegmentationDataset, load_data, remap_labels
from model import load_model, move_model
from compute_weights import compute_class_weights
from transformers import SegformerImageProcessor
from PIL import Image

# Set up the processor with do_reduce_labels=False so it doesn't modify our labels.
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    do_reduce_labels=False
)

# Define the mapping from original pixel values to contiguous indices [0-5]
label_mapping = {29: 0, 76: 1, 150: 2, 179: 3, 226: 4, 255: 5}

def transform_fn(image, label):
    label_np = np.array(label).astype(np.int32)
    label_remapped = remap_labels(label_np, label_mapping)
    encoded_inputs = processor(image, segmentation_maps=label_remapped, return_tensors="pt")
    return encoded_inputs

if __name__ == "__main__":
    # Set up base directory and load data
    base_dir = "/home/dquinteiro/Documentos/VSCodeProjects/SegFormer/ISPRS_Potsdam/ISPRS-Potsdam"
    train_df, validation_df = load_data(base_dir)

    # Create dataset and dataloader objects
    train_dataset = PotsdamSegmentationDataset(train_df, transform=transform_fn)
    valid_dataset = PotsdamSegmentationDataset(validation_df, transform=transform_fn)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Load global label counts from JSON (previously computed) and compute weights.
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_counts.json")
    num_labels = 6
    weights_tensor = compute_class_weights(json_path, label_mapping, num_labels)
    print("Computed weights tensor:", weights_tensor)
    
    # Build id2label and label2id based on sorted mapping keys.
    original_values = sorted(label_mapping.keys())  # [29, 76, 150, 179, 226, 255]
    id2label = {i: str(val) for i, val in enumerate(original_values)}
    label2id = {v: k for k, v in id2label.items()}

    # Load the model (this reinitializes the classifier head for 6 classes).
    model = load_model(num_labels, id2label, label2id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = move_model(model, device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Use our own weighted loss function.
    criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    num_epochs = 50
    checkpoint_interval = 10
    early_stopping_patience = 10
    best_val_miou = -1.0
    epochs_without_improvement = 0

    # Ensure the "metrics" folder exists for saving CSV files.
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        model.train()
        train_epoch_loss = 0.0

        # Training loop using our custom loss.
        for idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            pixel_values = batch["pixel_values"].to(device)  # shape: [batch, 3, H, W]
            labels = batch["labels"].to(device)              # shape: [batch, H, W]
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits  # shape: [batch, num_labels, H', W']

            # Upsample logits to match labels' spatial dimensions.
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            # Reshape logits to (N*H*W, num_labels) and labels to (N*H*W)
            loss = criterion(
                upsampled_logits.permute(0, 2, 3, 1).reshape(-1, num_labels),
                labels.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_epoch_loss = 0.0
        val_metric = evaluate.load("mean_iou")
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_dataloader, desc="Validation")):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = criterion(
                    upsampled_logits.permute(0, 2, 3, 1).reshape(-1, num_labels),
                    labels.reshape(-1)
                )
                val_epoch_loss += loss.item()
                
                predicted = upsampled_logits.argmax(dim=1)
                val_metric.add_batch(
                    predictions=predicted.detach().cpu().numpy(), 
                    references=labels.detach().cpu().numpy()
                )
        
        avg_val_loss = val_epoch_loss / len(valid_dataloader)
        computed_metrics = val_metric.compute(num_labels=num_labels, ignore_index=-100, reduce_labels=False)
        val_miou = computed_metrics.get("mean_iou", 0)
        val_acc = computed_metrics.get("mean_accuracy", 0)
        print(f"Validation Loss: {avg_val_loss:.4f} - mIoU: {val_miou:.4f} - Acc: {val_acc:.4f}")
        
        # Print per-class IoU if available (assuming computed_metrics["per_category_iou"] is an array)
        if "per_category_iou" in computed_metrics:
            print("Per-class IoU:")
            for i, iou in enumerate(computed_metrics["per_category_iou"]):
                print(f"  Class {i}: IoU = {iou:.4f}")

        # Checkpointing: save model every checkpoint_interval epochs.
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = f"segformer_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Early stopping based on validation mIoU.
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            epochs_without_improvement = 0
            best_model_path = "segformer_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best mIoU: {val_miou:.4f} -> saved best model checkpoint to {best_model_path}")
            # Save a CSV with the current best metrics.
            csv_path = os.path.join(metrics_dir, "best_metrics.csv")
            header = ["epoch", "mIoU", "accuracy"] + [f"class_{i}_iou" for i in range(num_labels)]
            row = [epoch+1, val_miou, val_acc] + list(computed_metrics.get("per_category_iou", []))
            with open(csv_path, "w", newline="") as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerow(row)
            print(f"Saved best metrics to {csv_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load the best model before saving final model.
    model.load_state_dict(torch.load("segformer_best.pt"))
    model.save_pretrained("segformer_potsdam_finetuned")
    print("Saved best model as final model in segformer_potsdam_finetuned")
