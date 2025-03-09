import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Normalize
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class PTDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        image = torch.load(file_path, weights_only=False)
        if self.transform:
            image = self.transform(image)
        return image, label

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def get_dataloader(files, labels, batch_size, num_workers, transform=None):
    dataset = PTDataset(files, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)

class ResNet3DForClassification(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3DForClassification, self).__init__()
        self.input_adapter = nn.Conv3d(1, 3, kernel_size=1)
        self.model = r3d_18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.input_adapter(x)
        return self.model(x)

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return roc_auc, sensitivity, specificity, accuracy, all_labels, all_probs, all_preds

def main():
    set_seed(42)
    transform = Compose([Normalize(mean=[0.5], std=[0.5])])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load .pt file paths and labels
    benign_files = [os.path.join('benign_converted', f) for f in os.listdir('benign_converted') if f.endswith('.pt')]
    malignant_files = [os.path.join('malign_original', f) for f in os.listdir('malign_original') if f.endswith('.pt')]

    benign_labels = [0] * len(benign_files)
    malignant_labels = [1] * len(malignant_files)

    all_files = benign_files + malignant_files
    all_labels = benign_labels + malignant_labels

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_val_labels = []
    all_val_probs = []
    all_val_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
        print(f"Evaluating Fold {fold + 1}...")

        val_files = [all_files[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]

        val_loader = get_dataloader(val_files, val_labels, batch_size=16, num_workers=8, transform=transform)

        # Load the corresponding model for this fold
        model_path = f"best_model_fold_{fold+1}.pth"
        model = ResNet3DForClassification(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path))  # Load fold-specific model

        roc_auc, sensitivity, specificity, accuracy, fold_val_labels, fold_val_probs, fold_val_preds = evaluate(model, val_loader, device)
        print(f"Fold {fold + 1}: ROC-AUC: {roc_auc:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}")

        fold_metrics.append({
            'Fold': fold + 1,
            'ROC-AUC': roc_auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Accuracy': accuracy
        })
        all_val_labels.extend(fold_val_labels)
        all_val_probs.extend(fold_val_probs)
        all_val_preds.extend(fold_val_preds)

    # Convert fold metrics to a DataFrame
    fold_metrics_df = pd.DataFrame(fold_metrics)
    print("\nFold Metrics:")
    print(fold_metrics_df)

    # Calculate mean metrics across all folds
    mean_metrics = fold_metrics_df[['ROC-AUC', 'Sensitivity', 'Specificity', 'Accuracy']].mean().to_dict()
    print("\nCross-Validation Results:")
    print(f"Mean ROC-AUC: {mean_metrics['ROC-AUC']:.4f}")
    print(f"Mean Sensitivity: {mean_metrics['Sensitivity']:.4f}")
    print(f"Mean Specificity: {mean_metrics['Specificity']:.4f}")
    print(f"Mean Accuracy: {mean_metrics['Accuracy']:.4f}")

    # Plot mean ROC curve
    fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs)
    roc_auc = roc_auc_score(all_val_labels, all_val_probs)
    cm = confusion_matrix(all_val_labels, all_val_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Mean ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Add mean metrics to the plot
    plt.text(0.6, 0.2, f'AUC: {roc_auc:.4f}', fontsize=10)
    plt.text(0.6, 0.15, f'Accuracy: {accuracy:.4f}', fontsize=10)
    plt.text(0.6, 0.1, f'Sensitivity: {sensitivity:.4f}', fontsize=10)
    plt.text(0.6, 0.05, f'Specificity: {specificity:.4f}', fontsize=10)

    plt.savefig('mean_roc_curve.png')  # Save mean ROC curve
    plt.show()

if __name__ == "__main__":
    main()