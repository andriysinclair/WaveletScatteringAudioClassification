# Libraries

import numpy as np

def apply_padding(data, desired_length_with_pad):
    """apply_padding

    Returns audio waves with 0 padding to match a desired length

    Args:
        data (dict): Dictionary with name of sound file (str) as key and sound wave (nd.array) as value
        desired_length_with_pad (int): Length for audio file to match after padding (should be set to maximum length of all sound waves in sample)

    Raises:
        ValueError: if the length of arrays after padding does not match the desired length of arrays

    Returns:
        dict: original sound wave dict with padding around audio file
    """    

    data_padded = {}
    test_set_padded = {}

    for k,v in data.items():

        # finding desired pad length
        pad_total = desired_length_with_pad - len(v)
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right

        v_padded = np.pad(v, pad_width=(pad_left, pad_right), mode="constant", constant_values = 0)
        print(v_padded[5000:7000])

        # Validate padded length
        if len(v_padded) != desired_length_with_pad:
            raise ValueError(f"Padded array for key '{k}' is {len(v_padded)} instead of {desired_length_with_pad}. Consider changing desired pad length")

        data_padded[k] = v_padded

    return data_padded