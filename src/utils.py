import numpy as np

def remap_labels(label_array, mapping):
    """
    Remap the values in label_array according to the provided dictionary 'mapping'.
    Converts the input to int32 and returns a new array.
    """
    label_array = label_array.astype(np.int32)
    remapped = label_array.copy()
    for orig_value, new_value in mapping.items():
        remapped[label_array == orig_value] = new_value
    return remapped
