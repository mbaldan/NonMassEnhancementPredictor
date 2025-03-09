import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import random
from torchvision.models.video import r3d_18  # Import 3D ResNet18
from torchvision.transforms import Compose, Normalize

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Options:
    def __init__(self):
        self.batch_size = 16
        self.num_workers = 8
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.patience = 5  # Early stopping patience

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
        image = torch.load(file_path, weights_only=False)  # Safe load
        if self.transform:
            image = self.transform(image)
        return image, label

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def prepare_datasets(train_files, train_labels, val_files, val_labels, transform=None):
    train_dataset = PTDataset(train_files, train_labels, transform=transform)
    val_dataset = PTDataset(val_files, val_labels, transform=transform)
    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)
    return train_loader, val_loader

class ResNet3DForClassification(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3DForClassification, self).__init__()
        self.input_adapter = nn.Conv3d(1, 3, kernel_size=1)
        self.model = r3d_18(weights="KINETICS400_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.input_adapter(x)
        return self.model(x)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    roc_auc = roc_auc_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return running_loss / total, correct / total, roc_auc, sensitivity, specificity

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    set_seed(42)

    opt = Options()
    transform = Compose([Normalize(mean=[0.5], std=[0.5])])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benign_files = [os.path.join('benign_converted', f) for f in os.listdir('benign_converted') if f.endswith('.pt')]
    malignant_files = [os.path.join('malign_converted', f) for f in os.listdir('malign_converted') if f.endswith('.pt')]

    benign_labels = [0] * len(benign_files)
    malignant_labels = [1] * len(malignant_files)

    all_files = benign_files + malignant_files
    all_labels = benign_labels + malignant_labels

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
        train_files = [all_files[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]

        train_dataset, val_dataset = prepare_datasets(train_files, train_labels, val_files, val_labels, transform=transform)
        train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, opt.batch_size, opt.num_workers)

        model = ResNet3DForClassification(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, opt.num_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, roc_auc, sensitivity, specificity = evaluate(model, val_loader, criterion, device)

            print(f"Fold {fold+1}, Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, ROC-AUC: {roc_auc:.4f}, "
                  f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

            # Save best model for each fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                fold_model_path = f"best_model_fold_{fold+1}.pth"
                torch.save(model.state_dict(), fold_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= opt.patience:
                print(f"Early stopping triggered for Fold {fold+1}. Best Val Loss: {best_val_loss:.4f}")
                break

if __name__ == "__main__":
    main()
