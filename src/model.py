import torch
from transformers import SegformerForSemanticSegmentation

def load_model(num_labels, id2label, label2id, checkpoint="nvidia/segformer-b0-finetuned-ade-512-512", ignore_mismatched_sizes=True):
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    return model

def move_model(model, device):
    model.to(device)
    return model
