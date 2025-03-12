import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torchvision import models
import math
from utils import preprocess_curves

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xavier_init(layer):
    if hasattr(layer, 'weight') and layer.weight.dim() > 1:
        nn.init.xavier_uniform_(layer.weight)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model: int, 
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = 4 * d_model

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return self.dropout(x)

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, input_size: int, bias: bool=True) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.embedding = nn.Linear(self.input_size, self.d_model, bias=bias)

    def forward(self, x) -> torch.Tensor:
        # (batch, seq_len, input_size) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float=0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, d_model: int, h: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        self.seq_len = seq_len
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False) # Wq
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False) # Wk
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False) # Wv
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False) # Wo
    
    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Ensure the mask has the correct shape
            mask = mask.unsqueeze(1)  # Add the num_heads dimension
            mask = mask.expand(-1, query.size(1), -1, -1)  # Expand to match attention scores
            # Apply mask
            attention_scores.masked_fill_(mask == 0, -65504.0)  # Minimum value for float16

        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax

        attention_scores = attention_scores

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        output, self.attention_scores = self.attention(query, key, value, mask)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, self.h * self.d_k)

        return self.w_o(output) # (B, 1, Dim) -> (B, 1, Dim)
    
class Encoder(nn.Module):

    def __init__(self, dim: int, n_heads: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.seq_len = seq_len
        
        self.attention = Attention(self.dim, self.n_heads, self.seq_len)
        self.feed_forward = FeedForwardBlock(self.dim)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(self.dim, eps=1e-5)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(self.dim, eps=1e-5)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:

        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        x_n = self.attention_norm(x)
        h = x + self.attention.forward(x_n, x_n, x_n, mask)

        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Decoder(nn.Module):

    def __init__(self, dim: int, n_heads: int, tgt_seq_len: int, custom_seq_len: int = 64, dropout: float = 0.0) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.tgt_seq_len = tgt_seq_len

        self.self_attention = Attention(self.dim, self.n_heads, self.tgt_seq_len)
        self.cross_attention = Attention(self.dim, self.n_heads, custom_seq_len)

        self.feed_forward = FeedForwardBlock(self.dim)

        # Normalization BEFORE the self attention block
        self.self_attention_norm = RMSNorm(self.dim, eps=1e-5)

        # Normalization BEFORE the self attention block
        self.cross_attention_norm = RMSNorm(self.dim, eps=1e-5)
        # Normalization BEFORE the self feed forward block
        self.cross_ffn_norm = RMSNorm(self.dim, eps=1e-5)

    def forward(self, 
                x: torch.Tensor, 
                encoder_output: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor,
                ) -> torch.Tensor:
        
        # Self-Attention with Residual Connection
        x_n = self.self_attention_norm(x)
        h = x + self.self_attention.forward(x_n, x_n, x_n, tgt_mask)

        # Cross-Attention with Residual Connection
        out_n = self.cross_attention_norm(h)
        h = h + self.cross_attention.forward(out_n, encoder_output, encoder_output, src_mask)

        # Feed-Forward with Residual Connection (single application)
        out = h + self.feed_forward.forward(self.cross_ffn_norm(h))

        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features

        self.layers = LinearLayer(self.in_features, self.out_features, True, False)

    def forward(self,x):
        x = self.layers(x)
        return x

class Contrastive_Curve(nn.Module):
    def __init__(self, in_channels=1, emb_size=128):
        super().__init__()
        self.convnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.convnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), bias=False, padding="same")
        # self.convnet.maxpool = Identity()
        self.convnet.fc = Identity()
        
        for p in self.convnet.parameters():
            p.requires_grad = True
            
        self.projector = ProjectionHead(2048, emb_size)

    def forward(self,x):
        
        x = torch.unsqueeze(x,1)

        out = self.convnet(x)
        
        xp = self.projector(torch.squeeze(out))
        
        return xp

class GraphHop(nn.Module):
    def __init__(self, num_node_features = 3, hidden_dim = 768, embedding_dim = 512, num_layers= 1, num_attn_heads=8):
        super(GraphHop, self).__init__()
        self.num_layers = num_layers
        self.projection = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList([
            gnn.GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )) for _ in range(num_layers)
        ])
        self.GAT = gnn.GATConv(hidden_dim * (num_layers+1), hidden_dim//4, heads=4)
        # self.residuals = nn.ModuleList([
        #     nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        # ])
        
        self.HopAttn = nn.MultiheadAttention(hidden_dim, num_attn_heads, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, target_idx = None):
        x = self.projection(x) # x: [num_nodes, hidden_dim]
        x_list = [x]

        for i in range(self.num_layers):
            h = self.convs[i](x_list[-1], edge_index) # h: [num_nodes, hidden_dim]
            # if i > 0:
            #     residual = self.residuals[i - 1](x_list[-1]) + h # residual: [num_nodes, hidden_dim]
            #     x_list.append(self.relu(residual)) # x_list[i]: [num_nodes, hidden_dim]
            # else:
            x_list.append(h) # x_list[i]: [num_nodes, hidden_dim]

        x = torch.cat(x_list, dim=1) # x: [num_nodes, hidden_dim * num_layers]
        x = self.GAT(x, edge_index) # x: [num_nodes, hidden_dim]
        x = self.relu(x) # x: [num_nodes, hidden_dim]
        
        Q = x.unsqueeze(1) # Q: [num_graphs, 1, hidden_dim]
        V = torch.cat([r.unsqueeze(1) for r in x_list], dim=1) # V: [num_graphs, num_layers+1, hidden_dim]
        K = V # K: [num_graphs, num_layers+1, hidden_dim]
        
        x = self.HopAttn(Q, K, V)[0].squeeze() # x: [num_graphs, hidden_dim]
        
        if target_idx is not None:
            x = x[target_idx] # x: [num_graphs, hidden_dim]
        else:
            x = gnn.global_mean_pool(x, batch) # x: [num_graphs, hidden_dim]
            
        x = self.fc(x) # x: [num_graphs, embedding_dim]
        return x
    
class Transformer(nn.Module):
    def __init__(self, tgt_seq_len: int, output_size: int, d_model: int=512, h: int=8, N: int=6):
        super(Transformer, self).__init__()

        # Positional Encoding for encoder and decoders
        self.encoder_positional_encoding = PositionalEncoding(d_model, seq_len=1)  # seq_len=1 for combined embeddings
        self.decoder_positional_encoding_first = PositionalEncoding(d_model, seq_len=tgt_seq_len // 2)
        self.decoder_positional_encoding_second = PositionalEncoding(d_model, seq_len=tgt_seq_len // 2)
        
        # Embeddings and Encoder
        self.tgt_embed = InputEmbeddings(d_model, output_size, bias=False)
        self.encoder = nn.ModuleList([Encoder(d_model, h, seq_len=1) for _ in range(N)])  # Input dim=2*d_model, seq_len=1

        # Two separate decoders
        self.decoder_first = nn.ModuleList([Decoder(d_model, h, tgt_seq_len // 2) for _ in range(N)])
        self.decoder_second = nn.ModuleList([Decoder(d_model, h, tgt_seq_len // 2) for _ in range(N)])

        # Projection layers for the decoders
        self.projection_first = nn.Linear(d_model, output_size, bias=False)
        self.projection_norm_first = RMSNorm(d_model, eps=1e-5)

        self.projection_second = nn.Linear(d_model, output_size, bias=False)
        self.projection_norm_second = RMSNorm(d_model, eps=1e-5)

        # Add Contrastive_Curve and GraphHop models
        self.contrastive_curve = Contrastive_Curve(in_channels=1, emb_size=d_model//2)
        self.graphhop = GraphHop(num_node_features=3, hidden_dim=d_model, embedding_dim=d_model//2)

        # Initialize weights
        self.apply(xavier_init)

    def encode(self, curve_data, graph_data, edge_index, batch):
        """
        Encode the input data using Contrastive_Curve, GraphHop, and the Transformer encoder.
        """
        curve_data = preprocess_curves(curves=curve_data)
        # Process curve data with Contrastive_Curve
        curve_embedding = self.contrastive_curve(curve_data).unsqueeze(0)  # Shape: (batch_size, d_model)
        # Process graph data with GraphHop
        graph_embedding = self.graphhop(graph_data, edge_index, batch)  # Shape: (batch_size, d_model)

        # Concatenate curve and graph embeddings
        combined_embedding = torch.cat([curve_embedding, graph_embedding], dim=-1)  # Shape: (batch_size, d_model)

        # Reshape combined_embedding to match encoder input shape (batch_size, seq_len, d_model)
        # Here, we treat the concatenated embeddings as a sequence of length 1
        combined_embedding = combined_embedding.unsqueeze(1)  # Shape: (batch_size, 1, d_model)

        # Add positional encoding to the combined embeddings
        src = self.encoder_positional_encoding(combined_embedding)  # Shape: (batch_size, 1, d_model)

        # Pass through encoder layers
        for layer in self.encoder:
            src = layer(src)  # Shape: (batch_size, 1, 2 * d_model)

        return src

    def decode(self, decoder_blocks, encoder_output, src_mask, tgt, tgt_mask, positional_encoding):
        """
        Apply the decoder layers to the target sequence.

        Args:
            decoder_blocks (nn.ModuleList): List of decoder layers.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask (not used in this case).
            tgt (torch.Tensor): Target sequence.
            tgt_mask (torch.Tensor): Target mask.
            positional_encoding (nn.Module): Positional encoding for the target sequence.

        Returns:
            torch.Tensor: Decoder output.
        """
        # Add positional encoding to the target sequence
        tgt = self.tgt_embed(tgt)  # Embed the target sequence
        tgt = positional_encoding(tgt)  # Add positional encoding

        # Apply decoder layers
        for layer in decoder_blocks:
            tgt = layer(tgt, encoder_output, src_mask, tgt_mask)

        return tgt
    
    def forward(self, decoder_input_first, decoder_input_second, mask_first, mask_second, curve_data, graph_data, edge_index, batch):
        # Encode source input with curve and graph data
        encoder_output = self.encode(curve_data, graph_data, edge_index, batch)

        # First decoder pass
        decoder_output_first = self.decode(
            self.decoder_first, encoder_output, None, decoder_input_first, mask_first, self.decoder_positional_encoding_first
        )
        final_output_first = self.projection_first(self.projection_norm_first(decoder_output_first))

        # Second decoder pass
        decoder_output_second = self.decode(
            self.decoder_second, encoder_output, None, decoder_input_second, mask_second, self.decoder_positional_encoding_second
        )
        final_output_second = self.projection_second(self.projection_norm_second(decoder_output_second))

        return final_output_first, final_output_second
