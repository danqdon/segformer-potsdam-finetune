import json
import torch

def compute_class_weights(json_path, label_mapping, num_labels):
    """
    Computes class weights based on global pixel counts.
    
    Args:
        json_path (str): Path to the global counts JSON file.
        label_mapping (dict): Mapping from original pixel values to class indices.
        num_labels (int): Total number of classes.
    
    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    with open(json_path, "r") as f:
        global_counts = json.load(f)
    
    # Build counts dictionary using remapped indices.
    # The keys in global_counts are strings, so we convert them to int.
    counts = {}
    for orig_str, count in global_counts.items():
        orig_val = int(orig_str)
        if orig_val in label_mapping:
            new_index = label_mapping[orig_val]
            counts[new_index] = count
        else:
            print(f"Warning: Original value {orig_val} not found in mapping.")
    
    total_pixels = sum(counts.values())
    
    # Compute weight for each class.
    weights = []
    for i in range(num_labels):
        count_i = counts.get(i, 1)  # avoid division by zero; ideally every class exists
        weight = total_pixels / (num_labels * count_i)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float)

if __name__ == "__main__":
    # Define your mapping and number of labels.
    label_mapping = {29: 0, 76: 1, 150: 2, 179: 3, 226: 4, 255: 5}
    num_labels = 6
    # Path to your global counts JSON file saved in src folder.
    json_path = "src/global_counts.json"
    
    weights_tensor = compute_class_weights(json_path, label_mapping, num_labels)
    print("Computed class weights tensor:", weights_tensor)
