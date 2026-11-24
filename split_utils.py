
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )

    # Group acyclic molecules under one fallback group.
    if scaffold == "":
        scaffold = "ACYCLIC"

    return scaffold

def scaffold_split(
    smiles_list,
    fractions=(0.8, 0.1, 0.1),
    seed=42
):

    assert abs(sum(fractions) - 1.0) < 1e-8

    scaffold_groups = {}

    for idx, smiles in enumerate(smiles_list):

        scaffold = get_scaffold(smiles)

        if scaffold is None:
            raise ValueError(
                f"Invalid SMILES at index {idx}: {smiles}"
            )

        scaffold_groups.setdefault(scaffold, []).append(idx)

    groups = list(scaffold_groups.values())

    # Deterministic shuffle to break equal-size group ties.
    rng = random.Random(seed)
    rng.shuffle(groups)

    # Larger groups are allocated first.
    groups.sort(key=len, reverse=True)

    n = len(smiles_list)

    targets = [
        fractions[0] * n,
        fractions[1] * n,
        fractions[2] * n
    ]

    assigned = [[], [], []]

    for group in groups:

        # Choose the split whose relative deficit is largest.
        deficits = [
            targets[i] - len(assigned[i])
            for i in range(3)
        ]

        split_idx = max(
            range(3),
            key=lambda i: deficits[i]
        )

        assigned[split_idx].extend(group)

    train_idx, val_idx, test_idx = assigned

    return train_idx, val_idx, test_idx

def verify_scaffold_split(smiles_list, splits):

    scaffold_sets = []

    for indices in splits:

        scaffolds = {
            get_scaffold(smiles_list[i])
            for i in indices
        }

        scaffold_sets.append(scaffolds)

    train_scaffolds = scaffold_sets[0]
    val_scaffolds = scaffold_sets[1]
    test_scaffolds = scaffold_sets[2]

    print("Train scaffolds:", len(train_scaffolds))
    print("Validation scaffolds:", len(val_scaffolds))
    print("Test scaffolds:", len(test_scaffolds))

    print(
        "Train/validation overlap:",
        len(train_scaffolds & val_scaffolds)
    )

    print(
        "Train/test overlap:",
        len(train_scaffolds & test_scaffolds)
    )

    print(
        "Validation/test overlap:",
        len(val_scaffolds & test_scaffolds)
    )

    assert not (
        train_scaffolds & val_scaffolds
    )

    assert not (
        train_scaffolds & test_scaffolds
    )

    assert not (
        val_scaffolds & test_scaffolds
    )

    print("Scaffold leakage check passed.")