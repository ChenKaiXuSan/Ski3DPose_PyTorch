# pure_stgcn.py

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer("A", A)
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (B, T, J, C)
        x = torch.einsum("btjc,jk->btkc", x, self.A)
        x = self.proj(x)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, dropout=0.1):
        super().__init__()

        self.gcn = GraphConv(in_channels, out_channels, A)

        padding = kernel_size // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, T, J, C)
        res = self.residual(x)

        x = self.gcn(x)  # (B, T, J, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, T, J)
        x = self.tcn(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, J, C)

        x = x + res
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    def __init__(
        self,
        num_joints,
        in_channels=3,
        hidden_channels=(64, 64, 128, 128, 256),
        edges=None,
        dropout=0.1,
    ):
        super().__init__()

        A = self.build_adjacency(num_joints, edges)
        self.register_buffer("A", A)

        channels = [in_channels] + list(hidden_channels)

        self.blocks = nn.ModuleList(
            [
                STGCNBlock(
                    channels[i],
                    channels[i + 1],
                    A,
                    dropout=dropout,
                )
                for i in range(len(hidden_channels))
            ]
        )

    @staticmethod
    def build_adjacency(num_joints, edges):
        A = torch.eye(num_joints)

        if edges is not None:
            for i, j in edges:
                A[i, j] = 1.0
                A[j, i] = 1.0

        D = A.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        A = D_inv_sqrt @ A @ D_inv_sqrt
        return A

    def forward(self, x, return_features=True):
        """
        x: (B, T, J, 3)
        """
        features = []

        for block in self.blocks:
            x = block(x)
            features.append(x)

        if return_features:
            return x, features
        return x


if __name__ == "__main__":
    B, T, J, C = 2, 100, 15, 3

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (1, 4),
        (4, 5),
        (5, 6),
        (1, 7),
        (7, 8),
        (8, 9),
        (7, 10),
        (10, 11),
        (11, 12),
        (7, 13),
        (13, 14),
    ]

    model = STGCN(
        num_joints=J,
        in_channels=3,
        hidden_channels=(64, 64, 128, 128, 256),
        edges=edges,
    )

    x = torch.randn(B, T, J, C)

    out, feats = model(x)

    print("input:", x.shape)
    print("out:", out.shape)

    for i, f in enumerate(feats):
        print(f"feat {i}:", f.shape)
