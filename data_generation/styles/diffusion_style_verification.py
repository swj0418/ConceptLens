import torch
import os
import pickle

def verify_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Display file size in MB
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    print(f"Loading file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Verify contents
    if not isinstance(data, list):
        print("The file does not contain a list.")
        return

    if len(data) == 0:
        print("The list is empty.")
        return

    # Print summary of contents
    print(f"Number of entries: {len(data)}")
    print("Sample structure:")
    sample = data[0]
    if isinstance(sample, dict):
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: Tensor with shape {value.shape}")
            else:
                print(f"  - {key}: {type(value).__name__}")
    else:
        print(f"Unexpected sample type: {type(sample).__name__}")


if __name__ == '__main__':
    # Path to the consolidated file
    file_path = 'ldm-after-50_styles_200.pkl'

    verify_file(file_path)
