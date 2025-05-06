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

class MoEFeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        experts_used: int = 2,
        dropout: float = 0.1,
        bias: bool = False,
        force_expert_utilization: bool = True
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.experts_used = experts_used
        self.dropout = dropout
        self.force_expert_utilization = force_expert_utilization
        
        hidden_dim = 4 * d_model
        
        # Shared expert (always used)
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=bias),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model, bias=bias)
        )
        
        # Specialized experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=bias),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model, bias=bias)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Tracking variables
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0))
        self.current_batch_usage = None
        
        # Forced utilization parameters
        if force_expert_utilization:
            self.min_expert_fraction = 0.1  # Each expert should get at least 10% of tokens
            self.expert_usage_history = []
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.gate.weight, std=0.02)
        for expert in self.experts:
            nn.init.xavier_uniform_(expert[0].weight)
            nn.init.xavier_uniform_(expert[3].weight)
            if expert[0].bias is not None:
                nn.init.constant_(expert[0].bias, 0.)
            if expert[3].bias is not None:
                nn.init.constant_(expert[3].bias, 0.)
        nn.init.xavier_uniform_(self.shared_expert[0].weight)
        nn.init.xavier_uniform_(self.shared_expert[3].weight)
        if self.shared_expert[0].bias is not None:
            nn.init.constant_(self.shared_expert[0].bias, 0.)
        if self.shared_expert[3].bias is not None:
            nn.init.constant_(self.shared_expert[3].bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch*seq_len, d_model)
        
        # Shared expert path
        shared_output = self.shared_expert(x)
        
        # Expert gating
        gate_logits = self.gate(x_flat)  # (batch*seq_len, num_experts)
        
        # Add noise during training for exploration
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.01
            gate_logits = gate_logits + noise
        
        # Get top-k experts
        top_k_gates, top_k_indices = gate_logits.topk(self.experts_used, dim=1)
        top_k_gates = torch.softmax(top_k_gates, dim=1)
        
        # Forced expert utilization
        if self.training and self.force_expert_utilization:
            # Calculate current expert usage
            expert_counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
            expert_fraction = expert_counts.float() / top_k_indices.numel()
            
            # Identify underutilized experts
            underutilized = expert_fraction < self.min_expert_fraction
            if underutilized.any():
                # Calculate redistribution probability
                num_underutilized = underutilized.sum()
                total_redistribute = (self.min_expert_fraction - expert_fraction[underutilized]).sum()
                
                # Adjust gate logits to favor underutilized experts
                gate_logits[:, underutilized] += total_redistribute / num_underutilized
                
                # Recompute top-k with adjusted logits
                top_k_gates, top_k_indices = gate_logits.topk(self.experts_used, dim=1)
                top_k_gates = torch.softmax(top_k_gates, dim=1)
        
        # Track expert usage
        with torch.no_grad():
            self.current_batch_usage = torch.bincount(
                top_k_indices.flatten(),
                minlength=self.num_experts
            ).float()
            self.current_batch_usage /= top_k_indices.numel()  # Convert to percentages
            
            if self.training:
                self.expert_counts += self.current_batch_usage * x.size(0)
                self.total_samples += x.size(0)
                if self.force_expert_utilization:
                    self.expert_usage_history.append(self.current_batch_usage.cpu())
        
        # Zero out non-selected experts
        gates = torch.zeros_like(gate_logits)  # (batch*seq_len, num_experts)
        gates.scatter_(1, top_k_indices, top_k_gates)  # Set only top-k gates
        
        # Process through all experts
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        
        # Weighted combination of expert outputs
        selected_experts = (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)
        selected_experts = selected_experts.view(batch_size, seq_len, d_model)
        
        # Combine with shared expert
        return shared_output + selected_experts

    def get_expert_usage(self, mode='current'):
        """Returns expert usage statistics
        Args:
            mode: 'current' - last batch usage
                  'average' - running average across all batches
                  'history' - full history (if force_expert_utilization=True)
        """
        if mode == 'current':
            return self.current_batch_usage
        elif mode == 'average':
            return self.expert_counts / (self.total_samples + 1e-6)
        elif mode == 'history':
            if not self.force_expert_utilization:
                raise ValueError("History only available when force_expert_utilization=True")
            return torch.stack(self.expert_usage_history) if self.expert_usage_history else None
        else:
            raise ValueError(f"Unknown mode {mode}")
        
# Then modify the Decoder class to use MoE
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

        # Replace standard FFN with MoE FFN
        self.feed_forward = MoEFeedForwardBlock(
            d_model=dim,
            num_experts=8,  # Total experts
            experts_used=2,  # Experts used per token
            dropout=dropout,
            bias=False
        )

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

        # Feed-Forward with Residual Connection (now using MoE)
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
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale1 = nn.Parameter(torch.ones(hidden_dim))
        self.layer_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x) * self.layer_scale1
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x) * self.layer_scale2
        x = self.dropout(x)
        return self.fc_out(x)

class ContrastiveEncoder(nn.Module):
    def __init__(self, in_channels=1, emb_size=128):
        super().__init__()
        self.convnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.convnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convnet.fc = nn.Identity()
        self.projector = ProjectionHead(2048, 2048, emb_size)

    def forward(self, x):
        features = self.convnet(x)
        return self.projector(features)

class Transformer(nn.Module):
    def __init__(self, tgt_seq_len: int, output_size: int, d_model: int=1024, h: int=8, N: int=6):
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
        self.contrastive_curve = ContrastiveEncoder(in_channels=1, emb_size=d_model//2)
        self.contrastive_adj = ContrastiveEncoder(in_channels=1, emb_size=d_model//2)

        # Initialize weights
        self.apply(xavier_init)

    def encode(self, curve_data, adj_data):
        curve_data = preprocess_curves(curves=curve_data).unsqueeze(1)

        # Process curve data with Contrastive_Curve
        curve_embedding = self.contrastive_curve(curve_data)  # Shape: (batch_size, d_model)
        # Process graph data with GraphHop
        adj_embedding = self.contrastive_adj(adj_data)  # Shape: (batch_size, d_model)

        combined_embedding = torch.cat([curve_embedding, adj_embedding], dim=1).unsqueeze(1)

        # Add positional encoding to the combined embeddings
        src = self.encoder_positional_encoding(combined_embedding)  # Shape: (batch_size, 1, d_model)

        # Pass through encoder layers
        for layer in self.encoder:
            src = layer(src)  # Shape: (batch_size, 1, d_model)

        return src, curve_embedding, adj_embedding

    def decode(self, decoder_blocks, encoder_output, src_mask, tgt, tgt_mask, positional_encoding):
        # Add positional encoding to the target sequence
        tgt = self.tgt_embed(tgt)  # Embed the target sequence
        tgt = positional_encoding(tgt)  # Add positional encoding

        # Apply decoder layers
        for layer in decoder_blocks:
            tgt = layer(tgt, encoder_output, src_mask, tgt_mask)

        return tgt
    
    def forward(self, decoder_input_first, decoder_input_second, mask_first, mask_second, curve_data, adj_dta):
        # Encode source input with curve and graph data
        encoder_output, curve_embedding, adj_embedding = self.encode(curve_data, adj_dta)

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

        return final_output_first, final_output_second, curve_embedding, adj_embedding 
