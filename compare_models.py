
import pandas as pd
import matplotlib.pyplot as plt

# Load model metrics
gcn = pd.read_csv("results/gcn_metrics.csv")
rf = pd.read_csv("results/rf_metrics.csv")

# Combine metrics
comparison = pd.DataFrame({
    "Model": ["GCN", "Random Forest"],
    "MAE": [
        gcn["test_mae"].iloc[0],
        rf["test_mae"].iloc[0]
    ],
    "RMSE": [
        gcn["test_rmse"].iloc[0],
        rf["test_rmse"].iloc[0]
    ],
    "R2": [
        gcn["test_r2"].iloc[0],
        rf["test_r2"].iloc[0]
    ]
})

print("\nMODEL COMPARISON")
print("============================")
print(comparison)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

# MAE comparison
plt.figure(figsize=(7, 5))

plt.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.ylabel("Test MAE")
plt.title("GCN vs Random Forest: MAE")

plt.tight_layout()

plt.savefig(
    "results/model_comparison_mae.png",
    dpi=300
)

plt.close()

# RMSE comparison
plt.figure(figsize=(7, 5))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.ylabel("Test RMSE")
plt.title("GCN vs Random Forest: RMSE")

plt.tight_layout()

plt.savefig(
    "results/model_comparison_rmse.png",
    dpi=300
)

plt.close()

# R2 comparison
plt.figure(figsize=(7, 5))

plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.ylabel("Test R2")
plt.title("GCN vs Random Forest: R2")

plt.tight_layout()

plt.savefig(
    "results/model_comparison_r2.png",
    dpi=300
)

plt.close()

print("\nComparison plots saved.")