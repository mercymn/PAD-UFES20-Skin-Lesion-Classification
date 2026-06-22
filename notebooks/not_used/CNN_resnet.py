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

df=pd.read_pickle("../data/skin-lesion-data/default.pickle")
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
    Path("../data/skin-lesion-data/images/train")
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


#Define CNN model using transfer learning (ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 output classes: benign vs malignant
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#Training and evaluation functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

#Training loop
NUM_EPOCHS = 15
best_val_acc = 0.0

# for epoch in range(NUM_EPOCHS):
#     train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
#     val_loss, val_acc = evaluate(model, val_loader, criterion, device)

#     print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
#           f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
#           f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

#     # Save best model
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), "best_model_cnn_resnet.pth")
#         print("  Saved best model")


#Load best model and evaluate on test set
model.load_state_dict(torch.load("best_model_cnn_resnet.pth"))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nTest loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

#Detailed classification report
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.tolist())
        all_preds.extend(predicted.cpu().tolist())

# Print classification report and confusion matrix once after all predictions
report = classification_report(all_labels, all_preds, target_names=["benign", "malignant"])
print("\nClassification Report:\n", report)
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)
