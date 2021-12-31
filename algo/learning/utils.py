import torch
import numpy as np
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler, Normalizer


def get_available_device():
    r"""Return the available device."""
    # set device to cpu or cuda
    device = torch.device('cpu')

    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    return device

def normailize_data(data, method='standardize'):
    r"""Normailize the data."""
    if method == 'standardize':
        norm_data = StandardScaler().fit_transform(data).astype('float32')
    else:
        norm_data = Normalizer().fit_transform(data).astype('float32')
    return norm_data

def load_pyg_data_from_network(network, attr_types=['resource'], normailize_method='standardize',
                        normailize_nodes_data=False, normailize_edges_data=False, ):
    """Load data from network"""

    # edge index
    edge_index = np.array(list(network.edges),dtype=np.int64).T
    edge_index = torch.LongTensor(edge_index)
    # node data
    n_attrs = network.get_node_attrs(attr_types)
    node_data = np.array(network.get_node_attrs_data(n_attrs), dtype=np.float32).T
    if normailize_nodes_data:
        node_data = normailize_data(node_data, method=normailize_method)
    node_data = torch.tensor(node_data)
    # edge data
    e_attrs = network.get_edge_attrs(attr_types)
    edge_data = np.array(network.get_edge_attrs_data(e_attrs), dtype=np.float32).T
    if normailize_edges_data:
        edge_data = normailize_data(edge_data, method=normailize_method)
    edge_data = torch.tensor(edge_data)
    # pyg data
    data = Data(x=node_data, edge_index=edge_index, edge_attr=edge_data)
    return data

def load_pyg_batch_from_network_list(network_list):
    data_list = []
    for network in network_list:
        data = load_pyg_data_from_network(network)
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    return batch