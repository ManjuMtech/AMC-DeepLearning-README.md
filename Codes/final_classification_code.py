import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

# ================= Preprocessing =================
# Assumes CSV columns: Real_0,...,Real_511, Imag_0,...,Imag_511, Mod_BPSK, Mod_QPSK, Mod_16QAM, SNR, Signal_Count

MODULATIONS = ["BPSK", "QPSK", "16QAM"]
SIGNAL_LENGTH = 512

def preprocess_data(df):
    # Extract features: first 2*SIGNAL_LENGTH columns (Real & Imag parts)
    X = df.iloc[:, :2 * SIGNAL_LENGTH].values
    # Reshape: from (N, 1024) to (N, 512, 2)
    X = X.reshape(X.shape[0], 2, SIGNAL_LENGTH).transpose(0, 2, 1)
    # Extract multi-label targets: next len(MODULATIONS) columns
    y = df.iloc[:, 2 * SIGNAL_LENGTH: 2 * SIGNAL_LENGTH + len(MODULATIONS)].values
    return X, y

# Load datasets (make sure these CSV files exist in your working directory)
train_df = pd.read_csv("train_multi_signal_count_dataset.csv")
test_df = pd.read_csv("test_multi_signal_count_dataset.csv")
X_train, y_train = preprocess_data(train_df)
X_test, y_test = preprocess_data(test_df)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 2 * SIGNAL_LENGTH)).reshape(-1, SIGNAL_LENGTH, 2)
X_test = scaler.transform(X_test.reshape(-1, 2 * SIGNAL_LENGTH)).reshape(-1, SIGNAL_LENGTH, 2)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)  # multi-label targets (binary vectors)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# ================= CNN-Transformer Multi-Label Model =================
class CNNTransformerMultiLabel(nn.Module):
    def __init__(self):
        super(CNNTransformerMultiLabel, self).__init__()
        # Input: (batch, 512, 2) --> convert to (batch, channels, sequence_length)
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)  # Halve the sequence length

        # After two poolings: sequence length = 512/2/2 = 128
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Final classification layer: output raw logits for 3 modulations
        self.fc = nn.Linear(64, len(MODULATIONS))

    def forward(self, x):
        # x: (batch, 512, 2)
        x = x.permute(0, 2, 1)  # -> (batch, 2, 512)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (batch, 32, 256)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (batch, 64, 128)
        x = x.permute(0, 2, 1)  # -> (batch, 128, 64)
        x = self.transformer(x)  # -> (batch, 128, 64)
        x = x.mean(dim=1)        # Global average pooling -> (batch, 64)
        x = self.fc(x)           # -> (batch, 3) raw logits
        return x

# ================= Training and Evaluation =================
model = CNNTransformerMultiLabel()
criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
clip_value = 1.0
num_epochs = 20

# For early stopping
best_test_loss = float('inf')
patience = 5
counter = 0

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_targets = []
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)  # (batch, 3) raw logits
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()  # threshold predictions at 0.5
        all_train_preds.append(preds.cpu().numpy())
        all_train_targets.append(batch_y.cpu().numpy())
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    all_train_preds = np.concatenate(all_train_preds, axis=0)
    all_train_targets = np.concatenate(all_train_targets, axis=0)
    train_acc = np.mean(all_train_preds == all_train_targets)  # element-wise accuracy
    train_accuracies.append(train_acc)

    model.eval()
    test_running_loss = 0.0
    all_test_preds = []
    all_test_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_test_preds.append(preds.cpu().numpy())
            all_test_targets.append(batch_y.cpu().numpy())
    test_loss = test_running_loss / len(test_loader)
    test_losses.append(test_loss)
    all_test_preds = np.concatenate(all_test_preds, axis=0)
    all_test_targets = np.concatenate(all_test_targets, axis=0)
    test_acc = np.mean(all_test_preds == all_test_targets)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Early Stopping Check
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        counter = 0
        # Optionally, save the model checkpoint here
        best_model = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    scheduler.step()

# Save plots 
os.makedirs("plots", exist_ok=True)
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, label="Train Loss", color='blue')
plt.plot(epochs_range, test_losses, label="Test Loss", color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs. Epochs")
plt.savefig("plots/loss_curve.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_accuracies, label="Train Accuracy", color='blue')
plt.plot(epochs_range, test_accuracies, label="Test Accuracy", color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs. Epochs")
plt.savefig("plots/accuracy_curve.png")
plt.show()

# ================= Detailed Evaluation Metrics =================
def evaluate_detailed(targets, preds):
    # Compute per-class precision, recall, and F1-score
    precision = precision_score(targets, preds, average=None, zero_division=0)
    recall = recall_score(targets, preds, average=None, zero_division=0)
    f1 = f1_score(targets, preds, average=None, zero_division=0)
    for i, mod in enumerate(MODULATIONS):
        print(f"{mod}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={f1[i]:.4f}")

    # Overall Signal Detection Metrics: A sample is considered signal-present if any label is 1.
    true_signal = (targets.sum(axis=1) > 0).astype(int)
    pred_signal = (preds.sum(axis=1) > 0).astype(int)
    overall_precision = precision_score(true_signal, pred_signal, average='binary', zero_division=0)
    overall_recall = recall_score(true_signal, pred_signal, average='binary', zero_division=0)
    overall_f1 = f1_score(true_signal, pred_signal, average='binary', zero_division=0)
    print(f"\nOverall Signal Detection - Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1-score: {overall_f1:.4f}")

print("\nDetailed Evaluation on Test Set:")
evaluate_detailed(all_test_targets, all_test_preds)

# ================= Per-Class Confusion Matrices =================
def plot_confusion_matrices(targets, preds, class_names):
    for i, cls in enumerate(class_names):
        cm = confusion_matrix(targets[:, i], preds[:, i])
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for {cls}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_{cls}.png")
        plt.show()

plot_confusion_matrices(all_test_targets, all_test_preds, MODULATIONS)
