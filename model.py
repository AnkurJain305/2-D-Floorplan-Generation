import sys
import types
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
class DummyPath:
    _path = []
sys.modules['torch.classes'] = types.SimpleNamespace(__path__=DummyPath())

# import torch
# import torch.nn as nn
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data, Batch
# from torch.utils.data import Dataset
# import torch_geometric.transforms as T
# from torch_geometric.nn import GATConv
# import torch.nn.functional as F
# from torch_geometric.nn import global_max_pool



# class GATNet(torch.nn.Module):
#     def __init__(self, num_graph_node_features, num_boundary_node_features):
#         super(GATNet, self).__init__()
        
#         # Convert input features to float32
#         self.register_buffer('dtype_template', torch.FloatTensor([1.0]))
        
#         # Graph convolutions
#         self.graph_conv1 = GATConv(num_graph_node_features, 32, heads=4)  # out: 128
#         self.graph_conv2 = GATConv(32*4, 32, heads=8)  # out: 256
#         self.graph_conv3 = GATConv(32*8, 64, heads=8)  # out: 512
#         self.graph_conv4 = GATConv(64*8, 128, heads=8)  # out: 1024
        
#         # Boundary convolutions
#         self.boundary_conv1 = GATConv(num_boundary_node_features, 32, heads=4)  # out: 128
#         self.boundary_conv2 = GATConv(128, 32, heads=8)  # out: 256
        
#         # After concatenating graph (1024) and boundary (256) features
#         total_features = 1024 + 256  # = 1280
#         self.Concatination1 = GATConv(total_features, 128, heads=8)  # out: 1024
        
#         # Linear layers
#         self.width_layer1 = nn.Linear(1024, 128)
#         self.height_layer1 = nn.Linear(1024, 128)
        
#         self.width_output = nn.Linear(128, 1)
#         self.height_output = nn.Linear(128, 1)
        
#         self.dropout = torch.nn.Dropout(0.2)
        
#     def forward(self, graph, boundary):
#         # Ensure input tensors are float32
#         x_graph = graph.x.type_as(self.dtype_template)
#         g_edge_index = graph.edge_index
#         g_edge_attr = graph.edge_attr.type_as(self.dtype_template) if graph.edge_attr is not None else None
#         g_batch = graph.batch
            
#         x_boundary = boundary.x.type_as(self.dtype_template)
#         b_edge_index = boundary.edge_index
#         b_edge_attr = boundary.edge_attr.type_as(self.dtype_template) if boundary.edge_attr is not None else None
#         b_batch = boundary.batch
            
#         NUM_OF_NODES = x_graph.shape[0]
            
#         if g_batch is None:
#             g_batch = torch.zeros(x_graph.shape[0], dtype=torch.long, device=x_graph.device)
#         if b_batch is None:
#             b_batch = torch.zeros(x_boundary.shape[0], dtype=torch.long, device=x_boundary.device)
            
#         # Graph branch
#         x_graph = F.leaky_relu(self.graph_conv1(x_graph, g_edge_index, g_edge_attr))
#         x_graph = self.dropout(x_graph)
            
#         x_graph = F.leaky_relu(self.graph_conv2(x_graph, g_edge_index, g_edge_attr))
#         x_graph = self.dropout(x_graph)
            
#         x_graph = F.leaky_relu(self.graph_conv3(x_graph, g_edge_index, g_edge_attr))
#         x_graph = self.dropout(x_graph)
            
#         x_graph = F.leaky_relu(self.graph_conv4(x_graph, g_edge_index, g_edge_attr))
#         x_graph = self.dropout(x_graph)
            
#         # Boundary branch
#         x_boundary = F.leaky_relu(self.boundary_conv1(x_boundary, b_edge_index, b_edge_attr))
#         x_boundary = self.dropout(x_boundary)
            
#         x_boundary = F.leaky_relu(self.boundary_conv2(x_boundary, b_edge_index, b_edge_attr))
#         x_boundary = self.dropout(x_boundary)
            
#         # Pool boundary features - this creates a single feature vector per graph
#         x_boundary_pooled = global_max_pool(x_boundary, b_batch)
            
#         # Expand boundary features to match graph nodes
#         # First, get the number of graphs in the batch
#         if g_batch.dim() == 1:
#             num_graphs = g_batch.max().item() + 1
#         else:
#             num_graphs = 1
                
#         # Repeat boundary features for each node in the corresponding graph
#         x_boundary_expanded = []
#         for i in range(num_graphs):
#             # Get nodes belonging to this graph
#             if num_graphs > 1:
#                 mask = (g_batch == i)
#                 num_nodes_in_graph = mask.sum()
#             else:
#                 num_nodes_in_graph = NUM_OF_NODES
                    
#             # Repeat boundary features for each node
#             graph_boundary_features = x_boundary_pooled[i:i+1].repeat(num_nodes_in_graph, 1)
#             x_boundary_expanded.append(graph_boundary_features)
            
#         x_boundary_expanded = torch.cat(x_boundary_expanded, dim=0)
            
#         # Concatenate features
#         x = torch.cat([x_graph, x_boundary_expanded], dim=1)
            
#         # Final convolution and predictions
#         x = F.leaky_relu(self.Concatination1(x, g_edge_index))
#         x = self.dropout(x)
            
#         width = F.relu(self.width_layer1(x))
#         width = self.dropout(width)
#         width = self.width_output(width)
            
#         height = F.relu(self.height_layer1(x))
#         height = self.dropout(height)

#         height = self.height_output(height)
        
#         return width.squeeze(), height.squeeze()
    
    
    
# def load_model(checkpoint_path, device):
#     model = GATNet(9, 3)
#     model = model.to(device)
    
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     return model

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
