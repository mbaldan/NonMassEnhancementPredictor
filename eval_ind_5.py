import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torchvision.models.video import r3d_18
from torchvision.transforms import Normalize
from monai.transforms import Compose, Resize, NormalizeIntensity, ToTensor

# Define transformations for NIfTI data (same as in training)
base_transforms = Compose([
    Resize((32, 128, 128)),  # Resize to (D, H, W)
    NormalizeIntensity(nonzero=True),  # Normalize based on nonzero values
    ToTensor()  # Convert to PyTorch tensor
])

# Define the model class
class ResNet3DForClassification(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3DForClassification, self).__init__()
        self.input_adapter = nn.Conv3d(1, 3, kernel_size=1)
        self.model = r3d_18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.input_adapter(x)
        return self.model(x)

# Function to preprocess and load a .nii file
def load_and_preprocess_nii(nii_file):
    try:
        # Load NIfTI file
        nii_data = nib.load(nii_file).get_fdata()
        # Add a channel dimension: (1, D, H, W)
        nii_data = np.expand_dims(nii_data, axis=0)
        # Apply transformations
        transformed_data = base_transforms(nii_data)

        # Ensure correct dimensions
        if transformed_data.ndim == 3:  # If shape is (D, H, W)
            transformed_data = transformed_data.unsqueeze(0)  # Add channel dimension

        if transformed_data.shape != (1, 32, 128, 128):
            raise ValueError(f"Unexpected tensor shape: {transformed_data.shape} for {nii_file}")

        print(f"Successfully preprocessed NIfTI file: {nii_file}")
        return transformed_data

    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the NIfTI file: {e}")

# Function to get model output for a specific patient using ensemble of 5 models
def get_ensemble_prediction(patient_data, model_paths, device):
    # Normalize patient data
    normalize = Normalize(mean=[0.5], std=[0.5])
    patient_data = normalize(patient_data)

    # Ensure patient data has the correct dimensions
    patient_data = patient_data.unsqueeze(0)  # Add batch dimension
    patient_data = patient_data.to(device, dtype=torch.float)

    # Load and evaluate each model
    all_probs = []
    for model_path in model_paths:
        model = ResNet3DForClassification(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            output = model(patient_data)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())  # Store probabilities

    # Convert to numpy array for averaging
    all_probs = np.array(all_probs)  # Shape: (5, 1, 2)

    # Average the probabilities across the 5 models
    mean_probs = np.mean(all_probs, axis=0)  # Shape: (1, 2)
    predicted_class = np.argmax(mean_probs)  # Get final predicted class

    return mean_probs, predicted_class

# Main function
def main():
    # Define paths
    nii_file = "aye kaldrmc 2.nii"  # Replace with the actual patient .nii file
    model_paths = [
        "best_model_fold_1.pth",
        "best_model_fold_2.pth",
        "best_model_fold_3.pth",
        "best_model_fold_4.pth",
        "best_model_fold_5.pth"
    ]  # List of all trained model files

    # Check if all models exist
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")

    # Load the processing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("All model files found. Device:", device)

    # Load and preprocess the NIfTI file
    patient_data = load_and_preprocess_nii(nii_file)

    # Get ensemble model output
    mean_probs, predicted_class = get_ensemble_prediction(patient_data, model_paths, device)

    print(f"Ensemble predicted probabilities: {mean_probs}")
    print(f"Final predicted class: {predicted_class}")

    if predicted_class == 0:
        print("Result: Benign")
    else:
        print("Result: Malignant")

if __name__ == "__main__":
    main()
