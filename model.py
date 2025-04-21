import sys
import types
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
class DummyPath:
    _path = []
sys.modules['torch.classes'] = types.SimpleNamespace(__path__=DummyPath())

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool
import torch.nn.functional as F

class GATNet(torch.nn.Module):
    def __init__(self, num_graph_node_features, num_boundary_node_features):
        super(GATNet, self).__init__()
        self.register_buffer('dtype_template', torch.FloatTensor([1.0]))

        self.graph_conv1 = GATConv(num_graph_node_features, 32, heads=4)
        self.graph_conv2 = GATConv(32*4, 32, heads=8)
        self.graph_conv3 = GATConv(32*8, 64, heads=8)
        self.graph_conv4 = GATConv(64*8, 128, heads=8)

        self.boundary_conv1 = GATConv(num_boundary_node_features, 32, heads=4)
        self.boundary_conv2 = GATConv(128, 32, heads=8)

        total_features = 1024 + 256
        self.Concatination1 = GATConv(total_features, 128, heads=8)

        self.width_layer1 = nn.Linear(1024, 128)
        self.height_layer1 = nn.Linear(1024, 128)

        self.width_output = nn.Linear(128, 1)
        self.height_output = nn.Linear(128, 1)

        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, graph, boundary):
        x_graph = graph.x.type_as(self.dtype_template)
        g_edge_index = graph.edge_index
        g_edge_attr = graph.edge_attr.type_as(self.dtype_template) if graph.edge_attr is not None else None
        g_batch = graph.batch or torch.zeros(x_graph.shape[0], dtype=torch.long, device=x_graph.device)

        x_boundary = boundary.x.type_as(self.dtype_template)
        b_edge_index = boundary.edge_index
        b_edge_attr = boundary.edge_attr.type_as(self.dtype_template) if boundary.edge_attr is not None else None
        b_batch = boundary.batch or torch.zeros(x_boundary.shape[0], dtype=torch.long, device=x_boundary.device)

        x_graph = F.leaky_relu(self.graph_conv1(x_graph, g_edge_index, g_edge_attr))
        x_graph = self.dropout(x_graph)
        x_graph = F.leaky_relu(self.graph_conv2(x_graph, g_edge_index, g_edge_attr))
        x_graph = self.dropout(x_graph)
        x_graph = F.leaky_relu(self.graph_conv3(x_graph, g_edge_index, g_edge_attr))
        x_graph = self.dropout(x_graph)
        x_graph = F.leaky_relu(self.graph_conv4(x_graph, g_edge_index, g_edge_attr))
        x_graph = self.dropout(x_graph)

        x_boundary = F.leaky_relu(self.boundary_conv1(x_boundary, b_edge_index, b_edge_attr))
        x_boundary = self.dropout(x_boundary)
        x_boundary = F.leaky_relu(self.boundary_conv2(x_boundary, b_edge_index, b_edge_attr))
        x_boundary = self.dropout(x_boundary)

        x_boundary_pooled = global_max_pool(x_boundary, b_batch)
        num_graphs = g_batch.max().item() + 1

        x_boundary_expanded = []
        for i in range(num_graphs):
            mask = (g_batch == i)
            num_nodes = mask.sum()
            expanded = x_boundary_pooled[i:i+1].repeat(num_nodes, 1)
            x_boundary_expanded.append(expanded)
        x_boundary_expanded = torch.cat(x_boundary_expanded, dim=0)

        x = torch.cat([x_graph, x_boundary_expanded], dim=1)
        x = F.leaky_relu(self.Concatination1(x, g_edge_index))
        x = self.dropout(x)

        width = self.width_output(self.dropout(F.relu(self.width_layer1(x))))
        height = self.height_output(self.dropout(F.relu(self.height_layer1(x))))

        return width.squeeze(), height.squeeze()

def load_model(checkpoint_path,device):
    model = GATNet(9, 3)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
