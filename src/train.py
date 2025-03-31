import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import numpy as np

from dataset import PotsdamSegmentationDataset, load_data, remap_labels
from model import load_model, move_model
from compute_weights import compute_class_weights
from transformers import SegformerImageProcessor
from PIL import Image

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    do_reduce_labels=False
)

label_mapping = {29: 0, 76: 1, 150: 2, 179: 3, 226: 4, 255: 5}

def transform_fn(image, label):
    label_np = np.array(label).astype(np.int32)
    label_remapped = remap_labels(label_np, label_mapping)
    encoded_inputs = processor(image, segmentation_maps=label_remapped, return_tensors="pt")
    return encoded_inputs

if __name__ == "__main__":
    base_dir = "/home/dquinteiro/Documentos/VSCodeProjects/SegFormer/ISPRS_Potsdam/ISPRS-Potsdam"
    train_df, validation_df = load_data(base_dir)

    train_dataset = PotsdamSegmentationDataset(train_df, transform=transform_fn)
    valid_dataset = PotsdamSegmentationDataset(validation_df, transform=transform_fn)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_counts.json")
    num_labels = 6
    weights_tensor = compute_class_weights(json_path, label_mapping, num_labels)
    print("Computed weights tensor:", weights_tensor)
    
    original_values = sorted(label_mapping.keys())  # [29, 76, 150, 179, 226, 255]
    id2label = {i: str(val) for i, val in enumerate(original_values)}
    label2id = {v: k for k, v in id2label.items()}

    model = load_model(num_labels, id2label, label2id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = move_model(model, device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    #TODO implement criterion into loss backward
    criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    num_epochs = 50
    checkpoint_interval = 10
    early_stopping_patience = 10
    best_val_miou = -1.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        model.train()
        train_epoch_loss = 0.0

        for idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            pixel_values = batch["pixel_values"].to(device)  # shape: [batch, 3, H, W]
            labels = batch["labels"].to(device)              # shape: [batch, H, W]
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_epoch_loss = 0.0
        val_metric = evaluate.load("mean_iou")
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_dataloader, desc="Validation")):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                val_epoch_loss += loss.item()
                
                logits = outputs.logits  # shape: [batch, num_labels, H', W']
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
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

        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = f"segformer_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            epochs_without_improvement = 0
            # Optionally, save the best model checkpoint here
            best_model_path = "segformer_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best mIoU: {val_miou:.4f} -> saved best model checkpoint to {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Save the final fine-tuned model
    model.save_pretrained("segformer_potsdam_finetuned")
