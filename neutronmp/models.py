import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class MetaLearner(nn.Module):
    def __init__(self, input_dim, num_classes, meta_hidden_dim=256, n_layers=3, dropout=0.3):
        super().__init__()
        layers = []
        in_features = input_dim
        for i in range(n_layers):
            out_features = meta_hidden_dim if i < n_layers - 1 else meta_hidden_dim // 2
            layers.append(nn.Linear(in_features, out_features))
            if i < n_layers - 1:
                layers.append(nn.LayerNorm(out_features))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_features = out_features
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(meta_hidden_dim // 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.mlp(x))


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.norms.append(nn.LayerNorm(hid_size))
        for _ in range(n_layers - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.norms.append(nn.LayerNorm(hid_size))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.norms.append(nn.LayerNorm(hid_size))
        self.dropout = nn.Dropout(0.5)
        self.input_proj = nn.Linear(in_size, hid_size) if in_size != hid_size else None

    def forward(self, blocks, x):
        h = x
        for i, (layer, block, norm) in enumerate(zip(self.layers, blocks, self.norms)):
            h_res = self.input_proj(h) if i == 0 and self.input_proj else h
            h = layer(block, h)
            if h.shape == h_res.shape:
                h = h + h_res
            if i != self.n_layers - 1:
                h = F.relu(h)
            h = norm(h)
            h = self.dropout(h)
        return h
