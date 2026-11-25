import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("results", exist_ok=True)


# ==========================================================
# LOAD PART 2 MODELS
# ==========================================================

files = []

if os.path.exists("results/rf_metrics.csv"):

    files.append(
        pd.read_csv("results/rf_metrics.csv")
    )


if os.path.exists("results/gcn_metrics.csv"):

    files.append(
        pd.read_csv("results/gcn_metrics.csv")
    )


# ==========================================================
# LOAD PART 3 MODELS
# ==========================================================

if os.path.exists("results/bond_gnn_comparison.csv"):

    bond_models = pd.read_csv(
        "results/bond_gnn_comparison.csv"
    )

    bond_models["model"] = (
        "BOND-AWARE "
        + bond_models["model"].astype(str).str.upper()
    )

    files.append(bond_models)


if len(files) == 0:

    raise FileNotFoundError(
        "No model metrics files were found."
    )


comparison = pd.concat(
    files,
    ignore_index=True
)


# ==========================================================
# STANDARDIZE MODEL NAMES
# ==========================================================

comparison["model"] = (
    comparison["model"]
    .astype(str)
    .str.upper()
)


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

comparison = comparison.drop_duplicates(
    subset=["model"],
    keep="last"
)


# ==========================================================
# SELECT COMPARISON COLUMNS
# ==========================================================

required_columns = [
    "model",
    "test_mae",
    "test_rmse",
    "test_r2"
]


for column in required_columns:

    if column not in comparison.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


comparison = comparison[
    required_columns
].copy()


# ==========================================================
# SORT BY MAE
# ==========================================================

comparison = comparison.sort_values(
    "test_mae"
)


comparison.to_csv(
    "results/part3_all_model_comparison.csv",
    index=False
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\nMODEL COMPARISON")

print("=" * 70)

print(comparison.to_string(index=False))


# ==========================================================
# MAE
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["model"],
    comparison["test_mae"]
)

plt.ylabel("Test MAE")

plt.xlabel("Model")

plt.title("Model Comparison - Test MAE")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "results/part3_comparison_mae.png",
    dpi=300
)

plt.close()


# ==========================================================
# RMSE
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["model"],
    comparison["test_rmse"]
)

plt.ylabel("Test RMSE")

plt.xlabel("Model")

plt.title("Model Comparison - Test RMSE")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "results/part3_comparison_rmse.png",
    dpi=300
)

plt.close()


# ==========================================================
# R2
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["model"],
    comparison["test_r2"]
)

plt.ylabel("Test R2")

plt.xlabel("Model")

plt.title("Model Comparison - Test R2")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "results/part3_comparison_r2.png",
    dpi=300
)

plt.close()


print("\nComparison files saved.")