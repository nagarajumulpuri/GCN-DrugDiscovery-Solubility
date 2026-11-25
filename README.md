# GCN Drug Discovery: Molecular Solubility Prediction
## Graph Neural Networks (GCN, GIN, GraphSAGE and GAT)
A step-by-step deep learning project for predicting molecular aqueous solubility (ESOL) using molecular graphs and Graph Neural Networks (GNNs). This project implements progressively more advanced molecular learning workflows, from a basic GCN to bond-aware GNN architectures, and compares their performance against a traditional Random Forest model.
## Project Overview
Molecular solubility is an important physicochemical property in drug discovery, influencing formulation, absorption, pharmacokinetics, screening and drug development.
In this project, molecular structures are represented as graphs, where atoms are nodes and chemical bonds are edges. Graph Neural Networks are trained to learn molecular representations and predict aqueous solubility.

The project is divided into three parts:
- Part 1: Build a basic Graph Convolutional Network (GCN) for molecular solubility prediction.
- Part 2: Improve the workflow using scaffold splitting, training-curve visualization, prediction analysis and Random Forest benchmarking.
- Part 3: Implement bond-aware GNNs and compare GCN, GIN, GraphSAGE and GAT architectures.

## Project Objectives
- Learn how molecular structures are represented as graphs.
- Implement Graph Convolutional Networks using PyTorch Geometric.
- Learn molecular representations using categorical atom features.
- Incorporate chemical bond features into graph neural networks.
- Implement GCN, GIN, GraphSAGE and GAT.
- Evaluate molecular property prediction models using MAE, RMSE and R².
- Perform scaffold-based data splitting to assess generalization to structurally distinct molecules.
- Compare GNNs with Random Forest using molecular fingerprints.
- Visualize training performance and predicted versus experimental values.
- Analyze prediction errors and model generalization.

## Dataset
## ESOL — Aqueous Solubility
The project uses the ESOL dataset available through PyTorch Geometric's MoleculeNet interface.

## Part 1 — Build a Basic GCN
## Objective
Develop a basic Graph Convolutional Network to predict molecular aqueous solubility from molecular graph representations.
## Workflow
1. Load ESOL dataset.
2. Convert molecules into graph representations.
3. Encode categorical atom features.
4. Apply graph convolution layers.
5. Perform graph-level pooling.
6. Predict molecular solubility.
7. Train using mean squared error.
8. Evaluate the model.

## GCN Architecture
The basic GCN consists of:
1. Atom embedding layers.
2. Graph convolution layer 1.
3. ReLU activation.
4. Graph convolution layer 2.
5. ReLU activation.
6. Global mean pooling.
7. Fully connected layers.
8. Solubility prediction.

## Architecture
Molecular SMILES <br>
      │  
      v  
Molecular Graph <br>
     |  <br>
     v  <br>
Categorical Atom Features <br>
     | <br>
     v <br>
Atom Embeddings <br>
     | <br>
     v <br>
GCN Layer 1  <br>
     | <br>
     v <br>
ReLU <br>
     | <br>
     v <br>
GCN Layer 2 <br>
     | <br>
     v <br>
ReLU <br>
     | <br>
     v <br>
Global Mean Pooling <br>
     | <br>
     v <br>
Fully Connected Layers <br>
     | <br>
     v <br>
Predicted Solubility <br>

## Training
The basic GCN is trained using:
 - Optimizer: Adam
 - Loss function: Mean Squared Error (MSE)
 - Activation: ReLU
 - Graph pooling: Global mean pooling
 - Output: Molecular solubility

## Evaluation metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

## Part 2 — Scaffold Split, Visualization and Random Forest Benchmarking
## Objective
Improve the basic GCN workflow and evaluate the model's ability to generalize to structurally distinct molecules.

### 2.1 Scaffold splitting
Random splitting may place structurally similar molecules into both training and test sets, potentially resulting in overly optimistic evaluation. <br>
To reduce this risk, this project uses a scaffold-based split using Bemis-Murcko scaffolds.
### Dataset split
 - Training: approximately 80%
 - Validation: approximately 10%
 - Test: approximately 10% <br>
The scaffold groups are assigned to separate partitions, with the split proportions approximated by a greedy allocation algorithm. <br>
The same split indices are reused for the Random Forest and subsequent GNN comparisons.
### Scaffold split objectives
 - Reduce structural overlap between partitions.
 - Evaluate model generalization to new molecular scaffolds.
 - Avoid using test data during model selection.
 - Ensure fair comparisons between models. <br>
The scaffold allocation implementation is a simplified greedy method, rather than a fully optimized scaffold-splitting algorithm. Acyclic molecules are grouped under a common fallback scaffold in the initial implementation, which can affect split proportions. This should be considered when interpreting the split.
### 2.2 GCN training improvements
The Part 2 GCN incorporates:
 - Scaffold splitting
 - Training and validation monitoring
 - Best-model checkpointing
 - Training-curve visualization
 - Prediction analysis
 - Test-set evaluation

### 2.3 Training curves
The training script generates:
 - Training MAE versus epoch
 - Validation MAE versus epoch
These curves help identify:
 - Model convergence
 - Overfitting
 - Underfitting
 - Training instability

### 2.4 Experimental versus predicted solubility
The project generates a scatter plot comparing:
 - Experimental solubility
 - Predicted solubility <br>
A diagonal reference line represents perfect prediction.

### 2.5 Prediction error analysis
The test predictions are analyzed using absolute error. <br>
The analysis can help identify:
 - Poorly predicted molecules
 - Unfamiliar scaffolds
 - Molecular complexity
 - Functional groups
 - Possible limitations in training data

### 2.6 Random Forest baseline
A Random Forest model is implemented using Morgan fingerprints.
### Molecular representation
 - Morgan fingerprint radius: 2
 - Fingerprint size: 2048 bits

### Random Forest configuration
 - Number of trees: 500
 - max_features: 0.7
 - min_samples_leaf: 1
 - Random seed: 42

### Workflow
1. Generate Morgan fingerprints.
2. Use the same scaffold split as the GCN.
3. Train the Random Forest.
4. Evaluate the model.
5. Save predictions.
6. Compare against the GCN.

### 2.7 Model comparison
The Part 2 comparison evaluates:
   - GCN
   - Random Forest
Metrics:
   - Test MAE
   - Test RMSE
   - Test R²

## Part 3 — Bond-Aware GNNs

