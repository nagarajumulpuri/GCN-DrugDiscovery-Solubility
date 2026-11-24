
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from torch_geometric.datasets import MoleculeNet

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. LOAD DATA
# ============================================================

dataset = MoleculeNet(
    root="data/ESOL",
    name="ESOL"
)

smiles_list = list(dataset.smiles)

targets = dataset.data.y.view(-1).numpy()

print("Dataset size:", len(smiles_list))

# ============================================================
# 2. LOAD THE SAME SPLIT
# ============================================================

with open("results/split_indices.json", "r") as f:

    split_indices = json.load(f)

train_idx = split_indices["train"]
val_idx = split_indices["validation"]
test_idx = split_indices["test"]

# ============================================================
# 3. GENERATE MORGAN FINGERPRINTS
# ============================================================

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

fingerprints = []

for smiles in smiles_list:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    fp = generator.GetFingerprint(mol)

    arr = np.zeros(
        (2048,),
        dtype=np.uint8
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr
    )

    fingerprints.append(arr)

X = np.array(fingerprints)

y = targets

# ============================================================
# 4. DATA SPLITS
# ============================================================

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(
    n_estimators=500,
    max_features=0.7,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

# ============================================================
# 6. VALIDATION EVALUATION
# ============================================================

val_predictions = rf.predict(X_val)

val_mae = mean_absolute_error(
    y_val,
    val_predictions
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_predictions
    )
)

val_r2 = r2_score(
    y_val,
    val_predictions
)

print("\nRF VALIDATION RESULTS")
print("-----------------------------")
print("Validation MAE:", val_mae)
print("Validation RMSE:", val_rmse)
print("Validation R2:", val_r2)

# ============================================================
# 7. TEST EVALUATION
# ============================================================

test_predictions = rf.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

print("\nRF TEST RESULTS")
print("-----------------------------")
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)
print("Test R2:", test_r2)

# ============================================================
# 8. SAVE METRICS
# ============================================================

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

metrics_df = pd.DataFrame([{
    "model": "RandomForest_Morgan",
    "validation_mae": val_mae,
    "validation_rmse": val_rmse,
    "validation_r2": val_r2,
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "test_r2": test_r2
}])

metrics_df.to_csv(
    "results/rf_metrics.csv",
    index=False
)

# ============================================================
# 9. SAVE TEST PREDICTIONS
# ============================================================

test_smiles = [
    smiles_list[i]
    for i in test_idx
]

predictions_df = pd.DataFrame({
    "SMILES": test_smiles,
    "Experimental": y_test,
    "Predicted": test_predictions,
    "Absolute_Error": np.abs(
        y_test - test_predictions
    )
})

predictions_df.to_csv(
    "results/rf_test_predictions.csv",
    index=False
)

# ============================================================
# 10. PREDICTION PLOT
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    test_predictions
)

low = min(
    y_test.min(),
    test_predictions.min()
)

high = max(
    y_test.max(),
    test_predictions.max()
)

plt.plot(
    [low, high],
    [low, high],
    linestyle="--"
)

plt.xlabel("Experimental Solubility")
plt.ylabel("Predicted Solubility")

plt.title(
    "Random Forest: Experimental vs Predicted"
)

plt.tight_layout()

plt.savefig(
    "results/rf_prediction_plot.png",
    dpi=300
)

plt.close()

# ============================================================
# 11. SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    rf,
    "models/random_forest.pkl"
)

print("\nRandom Forest results saved.")