pip install torch torchvision torchaudio
import sys
import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import streamlit as st
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
        nii_data = nib.load(nii_file).get_fdata()
        nii_data = np.expand_dims(nii_data, axis=0)  # Add channel dimension
        transformed_data = base_transforms(nii_data)

        if transformed_data.ndim == 3:  # If shape is (D, H, W)
            transformed_data = transformed_data.unsqueeze(0)  # Add channel dimension

        if transformed_data.shape != (1, 32, 128, 128):
            raise ValueError(f"Unexpected tensor shape: {transformed_data.shape} for {nii_file}")

        return transformed_data
    except Exception as e:
        st.error(f"Error processing NIfTI file: {e}")
        return None

# Function to get ensemble model prediction
def get_ensemble_prediction(patient_data, model_paths, device):
    normalize = Normalize(mean=[0.5], std=[0.5])
    patient_data = normalize(patient_data)
    patient_data = patient_data.unsqueeze(0).to(device, dtype=torch.float)

    all_probs = []
    for model_path in model_paths:
        model = ResNet3DForClassification(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            output = model(patient_data)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    mean_probs = np.mean(all_probs, axis=0)
    predicted_class = np.argmax(mean_probs)

    return mean_probs, predicted_class

# Streamlit App
st.title("Radiology AI Classifier")
st.write("Upload a **.nii** file to classify it as **Benign** or **Malignant**.")

# File uploader
uploaded_file = st.file_uploader("Choose a NIfTI (.nii) file", type=["nii"])

# Define model paths
model_paths = [
    "best_model_fold_1.pth",
    "best_model_fold_2.pth",
    "best_model_fold_3.pth",
    "best_model_fold_4.pth",
    "best_model_fold_5.pth"
]

# Check if all models exist
for model_path in model_paths:
    if not os.path.exists(model_path):
        st.error(f"Missing model file: {model_path}")
        st.stop()

# Processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_nii_path = "temp_uploaded.nii"
    with open(temp_nii_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully. Processing...")

    # Process file
    patient_data = load_and_preprocess_nii(temp_nii_path)
    
    if patient_data is not None:
        # Get model prediction
        mean_probs, predicted_class = get_ensemble_prediction(patient_data, model_paths, device)

        # Display results
        st.write("### Classification Result:")
        st.write(f"**Predicted Class:** {'Malignant' if predicted_class == 1 else 'Benign'}")
        st.write(f"**Confidence Scores:**")
        st.write(f"Benign: {mean_probs[0][0]:.4f}, Malignant: {mean_probs[0][1]:.4f}")

        # Remove temp file after processing
        os.remove(temp_nii_path)
