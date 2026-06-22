import pandas as pd
import numpy as np
import os, random, math, pathlib
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
print("PyTorch:", torch.__version__)

df=pd.read_pickle("C:/Users/vedha/OneDrive/Desktop/vs/group-coursework-meddm/data/skin-lesion-data/default.pickle")
df.head()

print(df['diagnostic'].unique())

df = df.copy()
df['diagnostic'] = df['diagnostic'].astype(str).str.upper()
malignant = ["BCC","MEL","SCC"]
df['label'] = df['diagnostic'].apply(lambda x: "malignant" if x in malignant else "benign")
df['label_num'] = df['label'].map({"benign":0, "malignant":1})
print(df[['img_id','diagnostic','label','label_num']].head())
print("\nCounts:\n", df['label'].value_counts())

from pathlib import Path

root_dirs = [
    Path("C:/Users/vedha/OneDrive/Desktop/vs/group-coursework-meddm/data/skin-lesion-data/images/train")
]

# Check again:
for p in root_dirs:
    print(p.resolve(), "-> exists:", p.exists())

# Step 2: test if metadata img_id values match actual image files
from collections import Counter

def find_file(fname):
    """Search for an image filename inside all 3 folders."""
    for p in root_dirs:
        candidate = p / fname
        if candidate.is_file():
            return candidate
    return None

missing = []
found = 0
n_test = min(200, len(df))  # check first 200 entries
for i in range(n_test):
    img_name = f"{df.iloc[i]['img_id']}.png"
    if find_file(img_name):
        found += 1
    else:
        missing.append(img_name)

print(f"Checked {n_test} rows — Found: {found}, Missing: {len(missing)}")
print("First 10 missing (if any):", missing[:10])

#Going through every image name in tabula dataset, look in all the 3 folders to check whether all 2298 images exist.
def find_file(fname):
    for p in root_dirs:#going through all the 3 folders
        candidate = p / fname
        if candidate.is_file():
            return candidate
    return None

missing = []
found = 0

for i in range(len(df)):# going through the entire dataframe
    img_name = str(df.iloc[i]["img_id"]).strip()
    if find_file(img_name):
        found += 1
    else:
        missing.append(img_name)

print(f"Checked {len(df)} rows — Found: {found}, Missing: {len(missing)}")

#Transformations; resizing and normalisation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),         # Resizing; 128x128 pixels
    transforms.RandomHorizontalFlip(),     # Flip images randomly (augmentation)
    transforms.RandomRotation(10),         # Rotate slightly+-10
    transforms.ToTensor(),                 # Convert from PIL image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalisation; using commonly used default values
                         [0.229, 0.224, 0.225])
])

# Validation/test sets — no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#Pytorch dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, df, root_dirs, transform=None):
        """
        df: metadata dataframe (includes img_id and label_num)
        root_dirs: list of image folders (imgs_part_1, imgs_part_2, imgs_part_3)
        transform: torchvision transforms to apply
        """
        self.df = df.reset_index(drop=True)  # store data
        self.root_dirs = root_dirs           # store list of folders
        self.transform = transform           # store transformations

    def __len__(self):
        """Return the number of samples (rows)"""
        return len(self.df)

    def _find_path(self, fname):
        """Find which folder contains the image file"""
        for rd in self.root_dirs:
            path = rd / fname
            if path.is_file():
                return path
        return None  # if not found

    def __getitem__(self, idx):
        """Return (image_tensor, label) for the idx-th sample"""
        row = self.df.iloc[idx]
        img_name = str(row['img_id']).strip()     # e.g., 'PAT_1516_1765_530.png'
        label = int(row['label_num'])             # 0 = benign, 1 = malignant

        path = self._find_path(img_name)
        if path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")

        # Open and process image
        img = Image.open(path).convert("RGB")     # ensures 3 color channels
        if self.transform:
            img = self.transform(img)             # apply preprocessing

        return img, label

#Patient-level split (no patient appears in more than one set)

#First split: 80% train and 20% temp
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df['patient_id']))
train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df  = df.iloc[temp_idx].reset_index(drop=True)

#Second split;spliting temp into 2; validation and testing set
gss2 = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df['patient_id']))
val_df = temp_df.iloc[val_idx_rel].reset_index(drop=True)
test_df = temp_df.iloc[test_idx_rel].reset_index(drop=True)

print("Train:", len(train_df), "Validation:", len(val_df), "Test:", len(test_df))

#Create datasets and dataloaders

train_dataset = SkinLesionDataset(train_df, root_dirs, transform=train_transform)
val_dataset   = SkinLesionDataset(val_df, root_dirs, transform=eval_transform)
test_dataset  = SkinLesionDataset(test_df, root_dirs, transform=eval_transform)
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

'''#Visualising images
import matplotlib.pyplot as plt
import torchvision

# get a batch of images and labels
images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
print("Labels:", labels[:8].tolist())

# unnormalize (reverse normalization for display)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
images_unnorm = images * std + mean

grid = torchvision.utils.make_grid(images_unnorm[:8], nrow=4)
plt.figure(figsize=(8,6))
plt.imshow(grid.permute(1,2,0).clip(0,1))
plt.title("Sample images (0=benign, 1=malignant)")
plt.axis("off")
plt.show()'''



# CNN MODEL DEFINITION

class SkinLesionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinLesionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # After 4 pooling operations: 128 -> 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# MODEL INITIALIZATION


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SkinLesionCNN(num_classes=2).to(device)
print(f"\nModel architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# LOSS FUNCTION AND OPTIMIZER


# Handle class imbalance with weighted loss
class_counts = df['label_num'].value_counts().sort_index()
total = len(df)
class_weights = torch.FloatTensor([total/class_counts[0], total/class_counts[1]]).to(device)
print(f"\nClass weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
factor=0.5, patience=3,)


# TRAINING AND VALIDATION FUNCTIONS


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# TRAINING LOOP


NUM_EPOCHS = 15
best_val_loss = float('inf')
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\n" + "="*60)
print("TRAINING STARTED")
print("="*60 + "\n")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_cnn_model.pth')
        print("  --> Best model saved!")
    print()

print("="*60)
print("TRAINING COMPLETED")
print("="*60 + "\n")

'''# PLOT TRAINING HISTORY
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(val_losses, label='Val Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(train_accs, label='Train Accuracy', marker='o')
ax2.plot(val_accs, label='Val Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()'''''

#EVALUATION ON TEST SET
# Load best model
model.load_state_dict(torch.load('best_cnn_model.pth'))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant

# Classification report
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60 + "\n")
print(classification_report(all_labels, all_preds,
                          target_names=['Benign', 'Malignant']))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)
'''''
# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign', 'Malignant'])
plt.yticks(tick_marks, ['Benign', 'Malignant'])

# Add text annotations
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = roc_auc_score(all_labels, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"\nAUC-ROC Score: {roc_auc:.4f}")'''
