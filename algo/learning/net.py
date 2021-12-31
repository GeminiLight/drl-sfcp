import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GatedGraphConv
from torch_geometric.data import Data, Batch


class MLPNet(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_layers=2):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim / 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_dim / 2, out_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNetBlock(nn.Module):
    
    def __init__(self, n_input_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, n_input_channels, kernel_size=[1, 1], stride=[1, 1])
        self.conv2 = nn.Conv2d(n_input_channels, n_input_channels, kernel_size=[1, 1], stride=[1, 1])

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out += identity
        return out


class GCNNet(nn.Module):
    r"""Graph Convolutional Network to extract the feature of physical network."""
    def __init__(self, input_dim, output_dim, embedding_dim=128, dropout_prob=0.2):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv3 = GCNConv(embedding_dim, output_dim)

    def forward(self, data):
        x, edge_index = data['x'], data['edge_index']
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = x.reshape(data.num_graphs, int(data.num_nodes/data.num_graphs), -1)
        return x


class GATNet(nn.Module):
    r"""Graph Attention Network to extract the feature of physical network."""
    def __init__(self, input_dim, output_dim, embedding_dim=128, heads=1, edge_dim=None, dropout_prob=0.2):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(input_dim, embedding_dim, heads=heads, edge_dim=edge_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv2 = GATConv(embedding_dim, embedding_dim, heads=heads, edge_dim=edge_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv3 = GATConv(embedding_dim, output_dim, heads=heads, edge_dim=edge_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data['x'], data['edge_index'], data.get('edge_attr', None)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = x.reshape(data.num_graphs, int(data.num_nodes/data.num_graphs), -1)
        return x
