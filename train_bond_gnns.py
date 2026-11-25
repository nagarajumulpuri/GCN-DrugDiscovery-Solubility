import os
import json
import random

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from bond_features import (
    load_esol_dataset,
    get_feature_sizes,
    print_dataset_information
)

from bond_aware_models import BondAwareGNN


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

EPOCHS = 200

BATCH_SIZE = 32

HIDDEN_DIM = 128

NUM_LAYERS = 3

LEARNING_RATE = 0.001

DROPOUT = 0.2

PATIENCE = 30

ARCHITECTURES = [
    "gcn",
    "gin",
    "graphsage",
    "gat"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================================
# RANDOM SEED
# ==========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# DATASET
# ==========================================================

dataset = load_esol_dataset()

print_dataset_information(dataset)

atom_feature_sizes, bond_feature_sizes = (
    get_feature_sizes(dataset)
)

print("\nAtom embedding sizes:")
print(atom_feature_sizes)

print("\nBond embedding sizes:")
print(bond_feature_sizes)

print("\nDevice:", DEVICE)


# ==========================================================
# LOAD PART 2 SPLIT
# ==========================================================

split_file = "results/split_indices.json"

if not os.path.exists(split_file):

    raise FileNotFoundError(
        "results/split_indices.json not found. "
        "Run Part 2 scaffold split first."
    )

with open(split_file, "r") as f:

    split_data = json.load(f)


train_idx = split_data["train"]

val_idx = split_data["validation"]

test_idx = split_data["test"]


print("\nSplit sizes")

print("Train:", len(train_idx))

print("Validation:", len(val_idx))

print("Test:", len(test_idx))


# ==========================================================
# CREATE SUBSETS
# ==========================================================

train_dataset = dataset[train_idx]

val_dataset = dataset[val_idx]

test_dataset = dataset[test_idx]


# ==========================================================
# DATA LOADERS
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_eval_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate(model, loader):

    model.eval()

    predictions = []

    actual_values = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            prediction = model(batch)

            actual = batch.y.view(-1)

            predictions.extend(
                prediction.cpu().numpy()
            )

            actual_values.extend(
                actual.cpu().numpy()
            )

    predictions = np.array(predictions)

    actual_values = np.array(actual_values)

    mae = mean_absolute_error(
        actual_values,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predictions
        )
    )

    r2 = r2_score(
        actual_values,
        predictions
    )

    return mae, rmse, r2, actual_values, predictions


# ==========================================================
# TRAIN ONE ARCHITECTURE
# ==========================================================

def train_model(architecture):

    print("\n" + "=" * 70)

    print(f"TRAINING {architecture.upper()}")

    print("=" * 70)


    model = BondAwareGNN(
        architecture=architecture,
        atom_feature_sizes=atom_feature_sizes,
        bond_feature_sizes=bond_feature_sizes,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )


    criterion = torch.nn.MSELoss()


    history = []

    best_val_mae = float("inf")

    best_epoch = 0

    patience_counter = 0

    best_model_path = (
        f"models/best_{architecture}_bond_gnn.pt"
    )


    # ------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0

        total_molecules = 0


        for batch in train_loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(batch)

            target = batch.y.view(-1)

            loss = criterion(
                prediction,
                target
            )

            loss.backward()

            optimizer.step()


            total_loss += (
                loss.item() * batch.num_graphs
            )

            total_molecules += batch.num_graphs


        train_loss = (
            total_loss / total_molecules
        )


        # --------------------------------------------------
        # EVALUATION
        # --------------------------------------------------

        train_mae, train_rmse, train_r2, _, _ = (
            evaluate(model, train_eval_loader)
        )

        val_mae, val_rmse, val_r2, _, _ = (
            evaluate(model, val_loader)
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


        # --------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------

        if val_mae < best_val_mae:

            best_val_mae = val_mae

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )

        else:

            patience_counter += 1


        # --------------------------------------------------
        # LOG
        # --------------------------------------------------

        if epoch == 1 or epoch % 10 == 0:

            print(

                f"Epoch {epoch:03d} | "

                f"Loss {train_loss:.4f} | "

                f"Train MAE {train_mae:.4f} | "

                f"Val MAE {val_mae:.4f} | "

                f"Val RMSE {val_rmse:.4f} | "

                f"Val R2 {val_r2:.4f}"

            )


        # --------------------------------------------------
        # EARLY STOPPING
        # --------------------------------------------------

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break


    # ======================================================
    # SAVE TRAINING HISTORY
    # ======================================================

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        f"results/{architecture}_bond_history.csv",
        index=False
    )


    # ======================================================
    # LOAD BEST MODEL
    # ======================================================

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )


    # ======================================================
    # FINAL TEST EVALUATION
    # ======================================================

    test_mae, test_rmse, test_r2, y_true, y_pred = (
        evaluate(model, test_loader)
    )


    # ======================================================
    # SAVE TEST PREDICTIONS
    # ======================================================

    smiles_test = [
        dataset.smiles[i]
        for i in test_idx
    ]


    prediction_df = pd.DataFrame({

        "SMILES": smiles_test,

        "Experimental": y_true,

        "Predicted": y_pred,

        "Absolute_Error": np.abs(
            y_true - y_pred
        )

    })


    prediction_df.to_csv(
        f"results/{architecture}_bond_predictions.csv",
        index=False
    )


    # ======================================================
    # PLOT TRAINING CURVES
    # ======================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_mae"],
        label="Train MAE"
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_mae"],
        label="Validation MAE"
    )

    plt.xlabel("Epoch")

    plt.ylabel("MAE")

    plt.title(
        f"{architecture.upper()} Training Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"results/{architecture}_bond_learning_curve.png",
        dpi=300
    )

    plt.close()


    # ======================================================
    # PREDICTION PLOT
    # ======================================================

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.7
    )


    min_value = min(
        y_true.min(),
        y_pred.min()
    )

    max_value = max(
        y_true.max(),
        y_pred.max()
    )


    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )


    plt.xlabel("Experimental Solubility")

    plt.ylabel("Predicted Solubility")

    plt.title(
        f"{architecture.upper()} Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        f"results/{architecture}_bond_prediction_plot.png",
        dpi=300
    )

    plt.close()


    # ======================================================
    # SAVE METRICS
    # ======================================================

    metrics = {

        "model": architecture.upper(),

        "best_epoch": best_epoch,

        "best_validation_mae": best_val_mae,

        "test_mae": test_mae,

        "test_rmse": test_rmse,

        "test_r2": test_r2

    }


    print("\nFinal test results")

    print("-------------------------------")

    print("Architecture:", architecture.upper())

    print("Best epoch:", best_epoch)

    print("Best validation MAE:", best_val_mae)

    print("Test MAE:", test_mae)

    print("Test RMSE:", test_rmse)

    print("Test R2:", test_r2)


    return metrics


# ==========================================================
# TRAIN ALL ARCHITECTURES
# ==========================================================

all_metrics = []


for architecture in ARCHITECTURES:

    metrics = train_model(
        architecture
    )

    all_metrics.append(
        metrics
    )


# ==========================================================
# SAVE COMPARISON RESULTS
# ==========================================================

comparison_df = pd.DataFrame(
    all_metrics
)

comparison_df.to_csv(
    "results/bond_gnn_comparison.csv",
    index=False
)


print("\n" + "=" * 70)

print("FINAL GNN COMPARISON")

print("=" * 70)


print(
    comparison_df.sort_values(
        "test_mae"
    )
)


print("\nAll models completed.")