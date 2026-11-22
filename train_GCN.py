
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# 2. LOAD DATASET
# ============================================================

dataset = MoleculeNet(
    root="data/ESOL",
    name="ESOL"
)

print("Dataset size:", len(dataset))
print("Example graph:", dataset[0])


# ============================================================
# 3. SCAFFOLD SPLIT
# ============================================================

# Obtain SMILES from the dataset object or graph objects.
if hasattr(dataset, "smiles"):
    smiles_list = list(dataset.smiles)
else:
    smiles_list = [data.smiles for data in dataset]

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )

    # Group acyclic compounds together by a fallback.
    if scaffold == "":
        scaffold = "ACYCLIC"

    return scaffold


scaffold_groups = {}

for idx, smiles in enumerate(smiles_list):
    scaffold = get_scaffold(smiles)

    if scaffold is None:
        continue

    scaffold_groups.setdefault(scaffold, []).append(idx)


# ============================================================
# 4. ASSIGN SCAFFOLD GROUPS TO TRAIN/VAL/TEST
# ============================================================

groups = list(scaffold_groups.values())

random.Random(SEED).shuffle(groups)

n_total = len(dataset)

train_target = int(0.80 * n_total)
val_target = int(0.10 * n_total)

train_indices = []
val_indices = []
test_indices = []

for group in groups:

    if len(train_indices) < train_target:
        train_indices.extend(group)

    elif len(val_indices) < val_target:
        val_indices.extend(group)

    else:
        test_indices.extend(group)


print("\nSplit sizes:")
print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))


# ============================================================
# 5. CREATE DATA LOADERS
# ============================================================

train_dataset = [dataset[i] for i in train_indices]
val_dataset = [dataset[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# 6. GCN MODEL
# ============================================================

class MolecularGCN(nn.Module):

    def __init__(self, feature_sizes, hidden_channels=128):
        super().__init__()

        # Each atom feature column is categorical.
        # Embed each column and sum the embeddings.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_channels)
            for size in feature_sizes
        ])

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

        self.dropout = 0.2

    def forward(self, x, edge_index, batch):

        # Embed categorical atom features.
        h = 0

        for column, embedding in enumerate(self.embeddings):
            h = h + embedding(x[:, column])

        # Graph convolution layer 1.
        h = self.conv1(h, edge_index)
        h = F.relu(h)

        # Graph convolution layer 2.
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        # Combine atom features into molecule features.
        h = global_mean_pool(h, batch)

        # Regression head.
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        return h.view(-1)


# ============================================================
# 7. INITIALIZE MODEL
# ============================================================

# Determine number of categories for each atom feature column.
feature_sizes = [
    int(dataset.data.x[:, i].max().item()) + 1
    for i in range(dataset.data.x.shape[1])
]

print("Atom feature category sizes:", feature_sizes)

model = MolecularGCN(
    feature_sizes=feature_sizes,
    hidden_channels=128
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

loss_function = nn.MSELoss()


# ============================================================
# 8. TRAINING FUNCTION
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

        loss = loss_function(predictions, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / total_graphs


# ============================================================
# 9. VALIDATION FUNCTION
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

        predictions.extend(output.cpu().numpy())
        targets.extend(batch.y.view(-1).cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = mean_absolute_error(targets, predictions)

    rmse = np.sqrt(
        mean_squared_error(targets, predictions)
    )

    r2 = r2_score(targets, predictions)

    return mae, rmse, r2, predictions, targets


# ============================================================
# 10. TRAIN MODEL
# ============================================================

EPOCHS = 200

best_val_mae = float("inf")
best_epoch = 0

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

for epoch in range(1, EPOCHS + 1):

    train_loss = train_epoch(train_loader)

    val_mae, val_rmse, val_r2, _, _ = evaluate(val_loader)

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
            f"Train MSE: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val R2: {val_r2:.4f}"
        )


# ============================================================
# 11. LOAD BEST MODEL
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
# 12. TEST EVALUATION
# ============================================================

test_mae, test_rmse, test_r2, test_predictions, test_targets = evaluate(
    test_loader
)

print("\nFINAL TEST RESULTS")
print("-----------------------------")
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)
print("Test R2:", test_r2)


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

test_smiles = [
    smiles_list[i] for i in test_indices
]

results_df = pd.DataFrame({
    "SMILES": test_smiles,
    "Experimental": test_targets,
    "Predicted": test_predictions,
    "Absolute_Error": np.abs(test_targets - test_predictions)
})

results_df.to_csv(
    "results/test_predictions.csv",
    index=False
)

print("\nSaved test predictions.")
