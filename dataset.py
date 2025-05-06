import numpy as np
import torch
from torch.utils.data import Dataset
import os

class TransformerDataset(Dataset):
    def __init__(self, data_dir):
        self.adj = np.load(os.path.join(data_dir, "adjacency.npy"), mmap_mode='r')
        self.curves = np.load(os.path.join(data_dir, "curves.npy"), mmap_mode='r')
        self.dec_in_first = np.load(os.path.join(data_dir, "decoder_input_first.npy"), mmap_mode='r')
        self.dec_in_second = np.load(os.path.join(data_dir, "decoder_input_second.npy"), mmap_mode='r')
        self.lbl_first = np.load(os.path.join(data_dir, "label_first.npy"), mmap_mode='r')
        self.lbl_second = np.load(os.path.join(data_dir, "label_second.npy"), mmap_mode='r')
        self.mask_first = np.load(os.path.join(data_dir, "mask_first.npy"), mmap_mode='r')
        self.mask_second = np.load(os.path.join(data_dir, "mask_second.npy"), mmap_mode='r')

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        return {
            "adjacency": torch.tensor(self.adj[idx], dtype=torch.float32),
            "curve_numerical": torch.tensor(self.curves[idx], dtype=torch.float32),
            "decoder_input_first": torch.tensor(self.dec_in_first[idx], dtype=torch.float32),
            "decoder_input_second": torch.tensor(self.dec_in_second[idx], dtype=torch.float32),
            "label_first": torch.tensor(self.lbl_first[idx], dtype=torch.float32),
            "label_second": torch.tensor(self.lbl_second[idx], dtype=torch.float32),
            "decoder_mask_first": torch.tensor(self.mask_first[idx], dtype=torch.bool),
            "decoder_mask_second": torch.tensor(self.mask_second[idx], dtype=torch.bool),
        }

# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class TransformerDataset(Dataset):
#     def __init__(self, node_features_path, edge_index_path, curves_path, max_nodes=20, shuffle=True):
#         self.node_features = np.load(node_features_path, allow_pickle=True)
#         self.edge_index = np.load(edge_index_path, allow_pickle=True)
#         self.curves = np.load(curves_path, mmap_mode='r')
#         self.max_nodes = max_nodes
#         self.max_sequence_length = 10  # for decoder sequence
#         if shuffle:
#             self._shuffle_data()

#     def _shuffle_data(self):
#         indices = np.random.permutation(len(self.node_features))
#         self.node_features = self.node_features[indices]
#         self.edge_index = self.edge_index[indices]
#         self.curves = self.curves[indices]

#     def __len__(self):
#         return len(self.node_features)

#     def __getitem__(self, idx):
#         raw_node = self.node_features[idx]             # shape: [n, 3]
#         edge_idx = self.edge_index[idx]                # shape: [2, num_edges]
#         curve = torch.tensor(self.curves[idx], dtype=torch.float32)  # [200, 2]

#         node_feats = raw_node[2:, :2]                  # shape: [n, 2]
#         node_attr = raw_node[:, 2]                     # shape: [n]
#         n = len(node_feats)

#         # Build adjacency
#         adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
#         for i, j in zip(edge_idx[0], edge_idx[1]):
#             adj[i, j] = 1.0
#             adj[j, i] = 1.0
#         for i in range(min(n, self.max_nodes)):
#             adj[i, i] = node_attr[i]
#         adjacency_tensor = torch.tensor(adj, dtype=torch.float32).unsqueeze(0)

#         # Split into two for decoder input
#         pos = torch.tensor(node_feats, dtype=torch.float32)
#         split_idx = pos.size(0) // 2
#         decoder_input_first, label_first = self._create_decoder_data(pos[:split_idx])
#         decoder_input_second, label_second = self._create_decoder_data(pos[split_idx:])
#         mask_first = self._create_combined_mask(decoder_input_first)
#         mask_second = self._create_combined_mask(decoder_input_second)

#         return {
#             "adjacency": adjacency_tensor,
#             "curve_numerical": curve,
#             "decoder_input_first": decoder_input_first,
#             "decoder_input_second": decoder_input_second,
#             "label_first": label_first,
#             "label_second": label_second,
#             "decoder_mask_first": mask_first,
#             "decoder_mask_second": mask_second,
#         }

#     def _create_decoder_data(self, pos):
#         # Insert <s> and pad up to max_sequence_length
#         num_nodes = pos.size(0)
#         pad_len = self.max_sequence_length - num_nodes - 1
#         decoder_input = torch.cat([
#             torch.ones(1, pos.size(1)) * -2.0,                  # <s>
#             pos,
#             torch.full((pad_len, pos.size(1)), -1.0)
#         ], dim=0)
#         label = torch.cat([
#             pos,
#             torch.ones(1, pos.size(1)),                         # </s>
#             torch.full((pad_len, pos.size(1)), -1.0)
#         ], dim=0)
#         return decoder_input, label

#     def _create_combined_mask(self, mech):
#         size = mech.size(0)
#         causal_mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
#         pad_mask = ~(mech == -1.0).all(dim=1)
#         pad_mask = pad_mask.unsqueeze(0).expand(size, -1)
#         combined_mask = causal_mask | ~pad_mask | ~pad_mask.T
#         return ~combined_mask
