import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load pickle data
def load_pickle_data(filepath):
    """Load data from pickle file"""
    with open("data/skin-lesion-data/numerical.pickle", 'rb') as f:
        data = pickle.load(f)
    return data

# Custom Dataset class
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# MLP Model (GEN AI)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super(MLPClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    return total_loss / len(train_loader), correct / total

# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return total_loss / len(val_loader), correct / total

# Main pipeline (GEN AI)
def main():
    # Configuration
    PICKLE_FILE = 'data/skin-lesion-data/numerical.pickle'  # Update with your file path
    HIDDEN_DIMS = [128, 64, 32]     # Architecture: 3 hidden layers
    DROPOUT = 0.3
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001

    # Load data
    print("Loading data...")
    data = load_pickle_data(PICKLE_FILE)

    # Assuming data is a dict with 'X' and 'y' keys
    # Adjust based on your pickle structure
    if isinstance(data, dict):
        X = data['X']
        y = data['y']
    elif isinstance(data, tuple):
        X, y = data
    else:
        # If data is a DataFrame
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Encode labels if they're strings
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=HIDDEN_DIMS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nTraining...")
    best_val_acc = 0
    patience = 15
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('C:/Users/vedha/OneDrive/Desktop/vs/mlp_cnn_classification_kit/best_model.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    # Print results
    print("\nTest Set Results:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Save the scaler for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\nModel and scaler saved!")

if __name__ == "__main__":
    main()
