import os
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, Resize, NormalizeIntensity, ToTensor

# Define base transformations for NIfTI data
base_transforms = Compose([
    Resize((32, 128, 128)),  # Resize to (D, H, W)
    NormalizeIntensity(nonzero=True),  # Normalize based on nonzero values
    ToTensor()  # Convert to PyTorch tensor
])

def process_and_save(input_dir, output_dir, transform):
    """
    Load .nii/.nii.gz files from the input directory, apply transformations,
    and save the processed tensors as .pt files in the output directory.

    Args:
        input_dir (str): Directory containing the input .nii/.nii.gz files.
        output_dir (str): Directory to save the processed .pt files.
        transform (callable): Transformations to apply to the data.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)

            try:
                # Load NIfTI file
                nii_data = nib.load(file_path).get_fdata()
                # Add a channel dimension: (1, D, H, W)
                nii_data = np.expand_dims(nii_data, axis=0)
                # Apply transformations
                transformed_data = transform(nii_data)

                # Add channel dimension if necessary
                if transformed_data.ndim == 3:  # If shape is (D, H, W)
                    transformed_data = transformed_data.unsqueeze(0)  # Add channel dimension

                # Validate shape
                if transformed_data.shape != (1, 32, 128, 128):  # Update based on model requirements
                    raise ValueError(f"Unexpected tensor shape: {transformed_data.shape} for {filename}")

                # Save as .pt file
                output_path = os.path.join(output_dir, filename.replace(".nii", ".pt").replace(".nii.gz", ".pt"))
                torch.save(transformed_data, output_path)

                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    # Input directories
    benign_input_dir = "benign"
    malign_input_dir = "malign"

    # Output directories
    benign_output_dir = "benign_converted"
    malign_output_dir = "malign_converted"

    # Process and save benign data
    print("Processing benign data...")
    process_and_save(benign_input_dir, benign_output_dir, base_transforms)

    # Process and save malign data
    print("Processing malign data...")
    process_and_save(malign_input_dir, malign_output_dir, base_transforms)

if __name__ == "__main__":
    main()
