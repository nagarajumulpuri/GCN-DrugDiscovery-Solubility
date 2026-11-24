import torch
from torch_geometric.datasets import MoleculeNet


def load_esol_dataset():
    dataset = MoleculeNet(
        root="data/ESOL",
        name="ESOL"
    )

    return dataset


def get_feature_sizes(dataset):
    atom_feature_sizes = []
    bond_feature_sizes = []

    for graph in dataset:
        if graph.x is not None:
            for column in range(graph.x.shape[1]):
                atom_feature_sizes.append(
                    int(graph.x[:, column].max().item()) + 1
                )

        if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
            for column in range(graph.edge_attr.shape[1]):
                bond_feature_sizes.append(
                    int(graph.edge_attr[:, column].max().item()) + 1
                )

        break

    return atom_feature_sizes, bond_feature_sizes


def print_dataset_information(dataset):

    graph = dataset[0]

    print("\nDataset information")
    print("---------------------------")
    print("Number of molecules:", len(dataset))
    print("Atoms in first molecule:", graph.x.shape[0])
    print("Atom feature dimensions:", graph.x.shape[1])
    print("Edges in first molecule:", graph.edge_index.shape[1])

    if graph.edge_attr is not None:
        print("Bond feature dimensions:", graph.edge_attr.shape[1])

    print("Target shape:", graph.y.shape)

    print("\nAtom features:")
    print(graph.x)

    print("\nEdge indices:")
    print(graph.edge_index)

    print("\nBond features:")
    print(graph.edge_attr)

    print("\nSolubility target:")
    print(graph.y)