# Libraries
import os
from pathlib import Path
from scipy.io import wavfile

# Test train spilt function based on index of audio file

def return_train_test_split(sound_files, index_to_test):

    max_file_length = 0

    # sort directory
    sound_files_sorted = sorted(sound_files.iterdir())

    # performing test train split
    train_set = {}
    test_set = {}

    for file in sound_files_sorted:

        # obtaining sound data
        sample_rate, x = wavfile.read(file)

        # finding max file length of dataset
        if len(x) > max_file_length:
            max_file_length = len(x)

        # extracting file name
        file_name = Path(file).stem
        
        # splitting at _
        file_index = file_name.split("_")
        file_index = int(file_index[-1])

        if file_index <= index_to_test:
            test_set[file_name] = x

        else:
            train_set[file_name] = x

    print(f"Longest sound wave is: {max_file_length}")

    return train_set, test_set
