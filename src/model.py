# src/model.py
import torch
from transformers import SegformerForSemanticSegmentation
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_segformer_model(num_labels=None, id2label_map=None, label2id_map=None,
                         model_checkpoint=None, ignore_classifier_size_mismatch=True):

    num_labels = num_labels if num_labels is not None else config.NUM_CLASSES
    id2label_map = id2label_map if id2label_map is not None else config.ID2LABEL_CONFIG_FOR_MODEL
    label2id_map = label2id_map if label2id_map is not None else config.LABEL2ID_CONFIG_FOR_MODEL
    model_checkpoint = model_checkpoint if model_checkpoint is not None else config.PRETRAINED_MODEL_CHECKPOINT

    logging.info(f"Loading SegFormer model from: {model_checkpoint}")
    logging.info(f"Configuring for {num_labels} labels with id2label: {id2label_map}")
    if ignore_classifier_size_mismatch:
        logging.info("Ignoring mismatched sizes for the classification head (fine-tuning).")

    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            id2label=id2label_map,
            label2id=label2id_map,
            ignore_mismatched_sizes=ignore_classifier_size_mismatch
        )
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_checkpoint}: {e}")
        raise

def move_model_to_device(model, target_device):
    try:
        model.to(target_device)
        logging.info(f"Model moved to device: {target_device}")
        return model
    except Exception as e:
        logging.error(f"Failed to move model to device {target_device}: {e}")
        raise