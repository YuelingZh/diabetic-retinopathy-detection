import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class ODIR_loader(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom dataset for ODIR data.

        Args:
            data_dir (str): Path to the directory containing .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            dict: A dictionary containing:
                - input (torch.Tensor): The image data.
                - label (torch.Tensor): The label (dr_class).
                - male (int): Male information (0 or 1).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)

        # Extract required fields
        image = data['slo_fundus']  # Image data
        label = data['dr_class']   # Label (dr_class)
        male = data['male']        # Male information

        # Convert image to torch tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]

        # Convert label and male to tensor
        label = torch.tensor(label, dtype=torch.long)  # For classification
        male = int(male)  # Keep as integer for grouping metrics

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        return {"input": image, "label": label, "male": male}

# if __name__ == "__main__":
#     # Define the dataset
#     dataset = ODIR_loader(data_dir="./ODIR_Data/test")
#
#     # Create a DataLoader
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
#
#     # Iterate through the dataset
#     for batch in data_loader:
#         print(batch)  # Each batch is a dictionary of tensors

# def debug_npz_file(file_path):
#     """
#     Debug the contents of an .npz file.
#
#     Args:
#         file_path (str): Path to the .npz file.
#     """
#     try:
#         data = np.load(file_path)
#         print(f"File: {file_path}")
#         print(f"Keys: {data.files}")  # Print all keys in the .npz file
#         for key in data.files:
#             value = data[key]
#             print(f"Key: {key}, Data Type: {value.dtype}, Shape: {value.shape}")
#             if value.ndim > 0:  # Only index if the data is not scalar
#                 print(f"First Few Values: {value[:5]}")  # Print first few values for inspection
#             else:
#                 print(f"Value: {value}")  # Print the scalar value directly
#         data.close()
#     except Exception as e:
#         print(f"Error loading file {file_path}: {e}")
#
#
# # Example usage
# if __name__ == "__main__":
#     directory = "./ODIR_Data/test"  # Adjust the directory path
#     npz_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
#
#     for npz_file in npz_files:
#         debug_npz_file(npz_file)
#         print("-" * 40)  # Separator for clarity