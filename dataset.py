import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    def __init__(self, node_features_path: str, edge_index_path: str, curves_path: str, shuffle=True):
        # Load the newly generated arrays
        self.node_features = np.load(node_features_path, allow_pickle=True).copy()
        self.edge_index = np.load(edge_index_path, allow_pickle=True).copy()
        
        # Load curves
        self.curves = np.load(curves_path, mmap_mode='r').copy()

        # Set max_sequence_length based on the maximum number of joints (20) + <s> and </s> tokens
        self.max_sequence_length = 7  # 20//2  joints + 1 <s> + 1 </s>
        
        # Shuffle the data if requested
        if shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # Generate shuffled indices
        indices = np.arange(len(self.node_features))
        np.random.shuffle(indices)
        
        # Apply shuffling to all arrays
        self.node_features = self.node_features[indices]
        self.edge_index = self.edge_index[indices]
        self.curves = self.curves[indices]

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        # Get the node features, edge index, num_joints, and curve for the given index
        node_features = self.node_features[idx]
        edge_index = self.edge_index[idx]
        curve = self.curves[idx]

        # Separate the first two columns (positional data) for the decoder
        pos = torch.tensor(node_features[:, :2], dtype=torch.float)  # Shape: [num_nodes, 2]

        # Use the remaining columns as node features for the encoder
        x = torch.tensor(node_features[:, 2:], dtype=torch.float)  # Shape: [num_nodes, num_node_features - 2]

        # Convert edge indices to a PyTorch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)  # Shape: [2, num_edges]

        # Convert the curve to a PyTorch tensor
        curve = torch.tensor(curve, dtype=torch.float)  # Shape: [200, 2]

        # Split the positional data into two halves
        split_idx = len(pos) // 2
        pos_first = pos[:split_idx]  # First half (x, y) pairs
        pos_second = pos[split_idx:]  # Second half (x, y) pairs

        # Create inputs and labels for the two decoders
        decoder_input_first, label_first = self._create_decoder_data(pos_first)
        decoder_input_second, label_second = self._create_decoder_data(pos_second)

        # Create causal masks for the two decoders 
        mask_first = self._create_combined_mask(decoder_input_first.size(0), decoder_input_first)
        mask_second = self._create_combined_mask(decoder_input_second.size(0), decoder_input_second)

        # Create a dictionary to hold the data
        data_dict = {
            "data": Data(x=x, edge_index=edge_index),  # Graph data wrapped in a PyG Data object
            "curve": curve,  # Target curve
            "decoder_input_first": decoder_input_first,  # Decoder input (first half)
            "decoder_input_second": decoder_input_second,  # Decoder input (second half)
            "label_first": label_first,  # Decoder labels (first half)
            "label_second": label_second,  # Decoder labels (second half)
            "decoder_mask_first": mask_first,  # Causal mask for the first decoder
            "decoder_mask_second": mask_second,  # Causal mask for the second decoder
        }

        return data_dict

    def _create_decoder_data(self, pos):
        """
        Prepare the decoder input and labels from the positional data.

        Args:
            pos (torch.Tensor): Positional data of shape [num_nodes, 2].

        Returns:
            decoder_input (torch.Tensor): Decoder input of shape [max_sequence_length, 2].
            label (torch.Tensor): Decoder labels of shape [max_sequence_length, 2].
        """
        num_nodes = pos.size(0)

        # Number of padding tokens required to reach `max_sequence_length`
        num_padding_tokens = self.max_sequence_length - num_nodes - 2  # Subtract sequence, <s>, and </s>

        # Decoder input: <s> token + sequence + padding
        decoder_input = torch.cat(
            [
                torch.zeros(1, pos.size(1), dtype=torch.float32),  # <s> token
                pos,  # Original sequence
                torch.ones(num_padding_tokens, pos.size(1), dtype=torch.float32) * -1,  # Padding (use -1 for consistency)
            ],
            dim=0,
        )

        # Label: sequence + </s> token + padding
        label = torch.cat(
            [
                pos,  # Original sequence
                torch.ones(1, pos.size(1), dtype=torch.float32),  # </s> token
                torch.ones(num_padding_tokens, pos.size(1), dtype=torch.float32) * -1,  # Padding (use -1 for consistency)
            ],
            dim=0,
        )

        return decoder_input, label

    def _create_combined_mask(self, size, mech):
        """
        Create a combined mask that includes both causal and padding masking.
        Ensures valid diagonal positions and excludes padding tokens.
        """
        # Create causal mask for the full size
        causal_mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.bool)  # Shape: (size, size)

        # Create padding mask: 1 for non-padding (valid positions), 0 for padding
        padding_mask = ~(mech == torch.tensor([-1.0, -1.0])).all(dim=1)  # Shape: (size,)
        padding_mask = padding_mask.unsqueeze(0).repeat(size, 1)  # Shape: (size, size)

        # Combine causal and padding masks
        combined_mask = causal_mask | ~padding_mask

        # Ensure rows and columns corresponding to padding tokens are entirely masked
        padding_mask_diag = padding_mask.clone()
        combined_mask = combined_mask | ~padding_mask_diag.T  # Mask padding rows completely

        # Flip the mask so True indicates allowed positions
        combined_mask = ~combined_mask

        return combined_mask  # True for allowed positions, False for disallowed