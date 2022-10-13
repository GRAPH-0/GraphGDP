import torch
import json
import os
import numpy as np
import os.path as osp
import pandas as pd
import pickle as pk

import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx, degree, to_networkx


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""

    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def networkx_graphs(dataset):
    return [to_networkx(dataset[i], to_undirected=True, remove_self_loops=True) for i in range(len(dataset))]


class StructureDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 dataset_name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.dataset_name = dataset_name

        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not os.path.exists(self.raw_paths[0]):
            raise ValueError("Without raw files.")
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_file_names(self):
        return [self.dataset_name + '.pkl']

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']

    @property
    def num_node_features(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.dataset_name}({arg_repr})'

    def process(self):
        # Read data into 'Data' list
        input_path = self.raw_paths[0]
        with open(input_path, 'rb') as f:
            graphs_nx = pk.load(f)
        data_list = [from_networkx(G) for G in graphs_nx]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    @torch.no_grad()
    def max_degree(self):
        data_list = [self.get(i) for i in range(len(self))]

        def graph_max_degree(g_data):
            return max(degree(g_data.edge_index[1], num_nodes=g_data.num_nodes))

        degree_list = [graph_max_degree(data) for data in data_list]
        return int(max(degree_list).item())

    def n_node_pmf(self):
        node_list = [self.get(i).num_nodes for i in range(len(self))]
        n_node_pmf = np.bincount(node_list)
        n_node_pmf = n_node_pmf / n_node_pmf.sum()
        return n_node_pmf


def get_dataset(config):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.

    Returns:
        train_ds, eval_ds, test_ds, n_node_pmf
    """
    # define data transforms
    transform = T.Compose([
        # T.ToDense(config.data.max_node),
        T.ToDevice(config.device)
    ])

    # Build up data iterators
    dataset = StructureDataset(config.data.root, config.data.name, transform=transform)
    num_train = int(len(dataset) * config.data.split_ratio)
    num_test = len(dataset) - num_train
    train_dataset = dataset[:num_train]
    eval_dataset = dataset[:num_test]
    test_dataset = dataset[num_train:]

    n_node_pmf = train_dataset.n_node_pmf()

    return train_dataset, eval_dataset, test_dataset, n_node_pmf
