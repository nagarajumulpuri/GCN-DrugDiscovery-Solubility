import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool


# ==========================================================
# 1. ATOM ENCODER
# ==========================================================

class AtomEncoder(nn.Module):

    def __init__(self, atom_feature_sizes, hidden_dim):

        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_dim)
            for size in atom_feature_sizes
        ])

    def forward(self, x):

        x = x.long()

        out = 0

        for column, embedding in enumerate(self.embeddings):
            out = out + embedding(x[:, column])

        return out


# ==========================================================
# 2. BOND ENCODER
# ==========================================================

class BondEncoder(nn.Module):

    def __init__(self, bond_feature_sizes, hidden_dim):

        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_dim)
            for size in bond_feature_sizes
        ])

    def forward(self, edge_attr):

        edge_attr = edge_attr.long()

        out = 0

        for column, embedding in enumerate(self.embeddings):
            out = out + embedding(edge_attr[:, column])

        return out


# ==========================================================
# 3. EDGE-AWARE GCN
# ==========================================================

class BondGCNLayer(MessagePassing):

    def __init__(self, hidden_dim):

        super().__init__(aggr="add")

        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_emb):

        num_nodes = x.size(0)

        # Add self-loops
        loop_index = torch.arange(
            num_nodes,
            device=x.device
        )

        self_edge_index = torch.stack(
            [loop_index, loop_index],
            dim=0
        )

        edge_index = torch.cat(
            [edge_index, self_edge_index],
            dim=1
        )

        # Self-loop bond embeddings are zero
        self_edge_emb = torch.zeros(
            num_nodes,
            edge_emb.size(1),
            device=x.device
        )

        edge_emb = torch.cat(
            [edge_emb, self_edge_emb],
            dim=0
        )

        row, col = edge_index

        # Symmetric degree normalization
        degree = torch.bincount(
            row,
            minlength=num_nodes
        ).float()

        degree_inv_sqrt = degree.pow(-0.5)

        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0

        norm = (
            degree_inv_sqrt[row]
            * degree_inv_sqrt[col]
        )

        x = self.linear(x)

        return self.propagate(
            edge_index,
            x=x,
            edge_emb=edge_emb,
            norm=norm
        )

    def message(self, x_j, edge_emb, norm):

        message = x_j + edge_emb

        return message * norm.unsqueeze(-1)


# ==========================================================
# 4. EDGE-AWARE GIN
# ==========================================================

class BondGINLayer(MessagePassing):

    def __init__(self, hidden_dim):

        super().__init__(aggr="add")

        self.epsilon = nn.Parameter(
            torch.zeros(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_emb):

        aggregated = self.propagate(
            edge_index,
            x=x,
            edge_emb=edge_emb
        )

        out = (1 + self.epsilon) * x + aggregated

        return self.mlp(out)

    def message(self, x_j, edge_emb):

        return x_j + edge_emb


# ==========================================================
# 5. EDGE-AWARE GRAPHSAGE
# ==========================================================

class BondSAGELayer(MessagePassing):

    def __init__(self, hidden_dim):

        super().__init__(aggr="mean")

        self.linear = nn.Linear(
            2 * hidden_dim,
            hidden_dim
        )

    def forward(self, x, edge_index, edge_emb):

        neighbor_aggregate = self.propagate(
            edge_index,
            x=x,
            edge_emb=edge_emb
        )

        combined = torch.cat(
            [x, neighbor_aggregate],
            dim=1
        )

        return self.linear(combined)

    def message(self, x_j, edge_emb):

        return x_j + edge_emb


# ==========================================================
# 6. EDGE-AWARE GAT
# ==========================================================

class BondGATLayer(MessagePassing):

    def __init__(self, hidden_dim, heads=4):

        super().__init__(
            aggr="add",
            node_dim=0
        )

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by heads"
            )

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.edge_linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.attention = nn.Parameter(
            torch.empty(
                heads,
                3 * self.head_dim
            )
        )

        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index, edge_emb):

        num_nodes = x.size(0)

        # Add self-loops
        loop_index = torch.arange(
            num_nodes,
            device=x.device
        )

        self_edge_index = torch.stack(
            [loop_index, loop_index],
            dim=0
        )

        edge_index = torch.cat(
            [edge_index, self_edge_index],
            dim=1
        )

        self_edge_emb = torch.zeros(
            num_nodes,
            edge_emb.size(1),
            device=x.device
        )

        edge_emb = torch.cat(
            [edge_emb, self_edge_emb],
            dim=0
        )

        x = self.linear(x)

        edge_emb = self.edge_linear(edge_emb)

        x = x.view(
            num_nodes,
            self.heads,
            self.head_dim
        )

        edge_emb = edge_emb.view(
            edge_emb.size(0),
            self.heads,
            self.head_dim
        )

        return self.propagate(
            edge_index,
            x=x,
            edge_emb=edge_emb
        )

    def message(
        self,
        x_i,
        x_j,
        edge_emb,
        index,
        ptr,
        size_i
    ):

        # Bond-enriched neighbor representation
        neighbor = x_j + edge_emb

        # Attention input:
        # receiver + neighbor + bond representation
        attention_input = torch.cat(
            [x_i, x_j, edge_emb],
            dim=-1
        )

        score = (
            attention_input
            * self.attention.unsqueeze(0)
        ).sum(dim=-1)

        score = F.leaky_relu(
            score,
            negative_slope=0.2
        )

        # Normalize across incoming neighbors
        alpha = torch_geometric_softmax(
            score,
            index
        )

        message = neighbor * alpha.unsqueeze(-1)

        return message.reshape(
            message.size(0),
            self.hidden_dim
        )


# ==========================================================
# 7. SOFTMAX FUNCTION FOR GAT
# ==========================================================

from torch_geometric.utils import softmax as torch_geometric_softmax


# ==========================================================
# 8. COMPLETE MODEL
# ==========================================================

class BondAwareGNN(nn.Module):

    def __init__(
        self,
        architecture,
        atom_feature_sizes,
        bond_feature_sizes,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2
    ):

        super().__init__()

        self.architecture = architecture.lower()

        self.atom_encoder = AtomEncoder(
            atom_feature_sizes,
            hidden_dim
        )

        self.bond_encoder = BondEncoder(
            bond_feature_sizes,
            hidden_dim
        )

        layer_classes = {
            "gcn": BondGCNLayer,
            "gin": BondGINLayer,
            "graphsage": BondSAGELayer,
            "gat": BondGATLayer
        }

        if self.architecture not in layer_classes:
            raise ValueError(
                f"Unknown architecture: {architecture}"
            )

        layer_class = layer_classes[self.architecture]

        self.layers = nn.ModuleList([
            layer_class(hidden_dim)
            if self.architecture != "gat"
            else layer_class(hidden_dim, heads=4)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = dropout

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):

        x = self.atom_encoder(data.x)

        edge_emb = self.bond_encoder(
            data.edge_attr
        )

        for layer, norm in zip(
            self.layers,
            self.norms
        ):

            residual = x

            x = layer(
                x,
                data.edge_index,
                edge_emb
            )

            x = norm(x)

            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training
            )

            x = x + residual

        graph_embedding = global_mean_pool(
            x,
            data.batch
        )

        prediction = self.regression_head(
            graph_embedding
        )

        return prediction.view(-1)