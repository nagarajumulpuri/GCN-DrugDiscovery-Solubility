
import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from split_utils import scaffold_split, verify_scaffold_split

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# 3. FOLDERS
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# 4. LOAD ESOL
# ============================================================

dataset = MoleculeNet(
    root="data/ESOL",
    name="ESOL"
)

smiles_list = list(dataset.smiles)

print("Dataset size:", len(dataset))

# ============================================================
# 5. SCAFFOLD SPLIT
# ============================================================

train_idx, val_idx, test_idx = scaffold_split(
    smiles_list,
    fractions=(0.8, 0.1, 0.1),
    seed=SEED
)

verify_scaffold_split(
    smiles_list,
    [train_idx, val_idx, test_idx]
)

print("\nSplit sizes:")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

# Save indices for reproducibility and fair comparison.
with open("results/split_indices.json", "w") as f:

    json.dump(
        {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx
        },
        f,
        indent=4
    )

# ============================================================
# 6. DATA LOADERS
# ============================================================

train_dataset = [dataset[i] for i in train_idx]
val_dataset = [dataset[i] for i in val_idx]
test_dataset = [dataset[i] for i in test_idx]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# 7. GCN ARCHITECTURE
# ============================================================

class MolecularGCN(nn.Module):

    def __init__(self, feature_sizes, hidden_channels=128):

        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_channels)
            for size in feature_sizes
        ])

        self.conv1 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.fc1 = nn.Linear(
            hidden_channels,
            hidden_channels
        )

        self.fc2 = nn.Linear(
            hidden_channels,
            1
        )

        self.dropout = 0.2

    def forward(self, x, edge_index, batch):

        h = 0

        for column, embedding in enumerate(self.embeddings):

            h = h + embedding(x[:, column])

        h = self.conv1(h, edge_index)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = F.relu(h)

        h = global_mean_pool(h, batch)

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training
        )

        h = F.relu(self.fc1(h))

        h = self.fc2(h)

        return h.view(-1)

# ============================================================
# 8. MODEL
# ============================================================

feature_sizes = [
    int(dataset.data.x[:, i].max().item()) + 1
    for i in range(dataset.data.x.shape[1])
]

model = MolecularGCN(
    feature_sizes,
    hidden_channels=HIDDEN_CHANNELS
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

loss_function = nn.MSELoss()

# ============================================================
# 9. TRAIN
# ============================================================

def train_epoch(loader):

    model.train()

    total_loss = 0
    total_graphs = 0

    for batch in loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        targets = batch.y.view(-1)

        loss = loss_function(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / total_graphs

# ============================================================
# 10. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    for batch in loader:

        batch = batch.to(DEVICE)

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        predictions.extend(
            output.cpu().numpy()
        )

        targets.extend(
            batch.y.view(-1).cpu().numpy()
        )

    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = mean_absolute_error(
        targets,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            targets,
            predictions
        )
    )

    r2 = r2_score(
        targets,
        predictions
    )

    return mae, rmse, r2, predictions, targets

# ============================================================
# 11. TRAINING LOOP + HISTORY
# ============================================================

history = []

best_val_mae = float("inf")
best_epoch = 0

for epoch in range(1, EPOCHS + 1):

    train_loss = train_epoch(train_loader)

    train_mae, train_rmse, train_r2, _, _ = evaluate(
        train_loader
    )

    val_mae, val_rmse, val_r2, _, _ = evaluate(
        val_loader
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

    if val_mae < best_val_mae:

        best_val_mae = val_mae
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            "models/best_gcn.pt"
        )

    if epoch % 10 == 0 or epoch == 1:

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train MAE {train_mae:.4f} | "
            f"Val MAE {val_mae:.4f} | "
            f"Val RMSE {val_rmse:.4f} | "
            f"Val R2 {val_r2:.4f}"
        )

# ============================================================
# 12. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    "results/training_history.csv",
    index=False
)

# ============================================================
# 13. PLOT TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_mae"],
    label="Training MAE"
)

plt.plot(
    history_df["epoch"],
    history_df["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("GCN Training and Validation MAE")
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/training_curves.png",
    dpi=300
)

plt.close()

# ============================================================
# 14. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        "models/best_gcn.pt",
        map_location=DEVICE
    )
)

print("\nBest epoch:", best_epoch)
print("Best validation MAE:", best_val_mae)

# ============================================================
# 15. TEST
# ============================================================

test_mae, test_rmse, test_r2, test_predictions, test_targets = evaluate(
    test_loader
)

print("\nFINAL GCN TEST RESULTS")
print("-----------------------------")
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)
print("Test R2:", test_r2)

# ============================================================
# 16. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{
    "model": "GCN",
    "best_epoch": best_epoch,
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "test_r2": test_r2
}])

metrics_df.to_csv(
    "results/gcn_metrics.csv",
    index=False
)

# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

test_smiles = [
    smiles_list[i]
    for i in test_idx
]

predictions_df = pd.DataFrame({
    "SMILES": test_smiles,
    "Experimental": test_targets,
    "Predicted": test_predictions,
    "Absolute_Error": np.abs(
        test_targets - test_predictions
    )
})

predictions_df.to_csv(
    "results/gcn_test_predictions.csv",
    index=False
)

# ============================================================
# 18. PREDICTION PLOT
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    test_targets,
    test_predictions
)

low = min(
    test_targets.min(),
    test_predictions.min()
)

high = max(
    test_targets.max(),
    test_predictions.max()
)

plt.plot(
    [low, high],
    [low, high],
    linestyle="--"
)

plt.xlabel("Experimental Solubility")
plt.ylabel("Predicted Solubility")
plt.title("GCN: Experimental vs Predicted")

plt.tight_layout()

plt.savefig(
    "results/gcn_prediction_plot.png",
    dpi=300
)

plt.close()

# ============================================================
# 19. ERROR ANALYSIS
# ============================================================

predictions_df = predictions_df.sort_values(
    "Absolute_Error",
    ascending=False
)

predictions_df.to_csv(
    "results/gcn_test_predictions_sorted.csv",
    index=False
)

print("\nLargest test errors:")
print(
    predictions_df.head(10)
)

print("\nAll results saved in results/")