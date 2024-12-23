import torch
import torch.nn as nn
import math
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import random

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

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
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
    def __init__(self, d_model: int, h: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        self.seq_len = seq_len
        self.dropout = dropout
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False) # Wq
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False) # Wk
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False) # Wv
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False) # Wo

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
    
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
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax

        attention_scores = self.attn_dropout(attention_scores)

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

        return self.resid_dropout(self.w_o(output)) # (B, 1, Dim) -> (B, 1, Dim)
    
class Encoder(nn.Module):

    def __init__(self, dim: int, n_heads: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.seq_len = seq_len
        
        self.attention = Attention(self.dim, self.n_heads, self.seq_len, self.dropout)
        self.feed_forward = FeedForwardBlock(self.dim)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(self.dim, eps=1e-5)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(self.dim, eps=1e-5)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

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

        self.self_attention = Attention(self.dim, self.n_heads, self.tgt_seq_len, self.dropout)
        self.cross_attention = Attention(self.dim, self.n_heads, custom_seq_len, self.dropout)

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
    
class Transformer(pl.LightningModule):
    def __init__(self, tgt_seq_len: int, output_size: int, d_model: int=512, h: int=8, N: int=12, num_mech_types: int=175):
        super(Transformer, self).__init__()

        # Model components
        num_patches = (64 // 8) ** 2
        self.patch_embedding = nn.Conv2d(1, d_model, kernel_size=8, stride=8)
        self.src_seq_len = num_patches + 1

        # Mechanism type embedding
        self.mech_type_embedding = nn.Linear(num_mech_types, d_model, bias=False)

        # Embeddings and Encoder
        self.tgt_embed = InputEmbeddings(d_model, output_size, bias=False)
        self.encoder = nn.ModuleList([Encoder(d_model, h, self.src_seq_len) for _ in range(N)])

        # Two separate decoders
        self.decoder_first = nn.ModuleList([Decoder(d_model, h, tgt_seq_len // 2) for _ in range(N)])
        self.decoder_second = nn.ModuleList([Decoder(d_model, h, tgt_seq_len // 2) for _ in range(N)])

        # Projection layers for the decoders
        self.projection_first = nn.Linear(d_model, output_size, bias=False)
        self.projection_norm_first = RMSNorm(d_model, eps=1e-5)

        self.projection_second = nn.Linear(d_model, output_size, bias=False)
        self.projection_norm_second = RMSNorm(d_model, eps=1e-5)

        # Initialize weights
        self.apply(xavier_init)

    def encode(self, src, src_mask, mech_type):
        # Compute mechanism type embedding
        mech_type_embed = self.mech_type_embedding(mech_type)  # Shape: (batch_size, d_model)

        # Expand and add mechanism type embedding to the patch embedding
        src = self.patch_embedding(src)  # Shape: (batch_size, seq_len, d_model)
        src = src.view(src.shape[0], -1, src.shape[-3])  # Flatten spatial dimensions

        # Add mechanism type embedding to each position in the sequence
        src = src + mech_type_embed.unsqueeze(1)  # Broadcasting over the sequence length

        # Pass through encoder layers
        for layer in self.encoder:
            src = layer(src, src_mask)
        return src

    def decode(self, decoder_blocks, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        for layer in decoder_blocks:
            tgt = layer(tgt, encoder_output, src_mask, tgt_mask)
        return tgt

    def forward(self, source, decoder_input_first, decoder_input_second, mask_first, mask_second, mech_type):
        # Encode source input with mechanism type
        encoder_output = self.encode(source, None, mech_type)

        # First decoder pass
        decoder_output_first = self.decode(self.decoder_first, encoder_output, None, decoder_input_first, mask_first)
        final_output_first = self.projection_first(self.projection_norm_first(decoder_output_first))

        # Second decoder pass
        decoder_output_second = self.decode(self.decoder_second, encoder_output, None, decoder_input_second, mask_second)
        final_output_second = self.projection_second(self.projection_norm_second(decoder_output_second))

        return final_output_first, final_output_second
    
    def mse_loss(self, predictions, targets, mask_value=0.5):
        """
        Computes the masked MSE loss.

        Args:
            predictions (torch.Tensor): Predicted outputs from the model (batch_size, seq_len, output_dim).
            targets (torch.Tensor): Ground truth labels (batch_size, seq_len, output_dim).
            mask_value (float): The value used for padding tokens (e.g., [0.5, 0.5]).

        Returns:
            torch.Tensor: The masked mean squared error loss.
        """
        # Create a mask where True indicates non-padding tokens
        mask = ~(targets == mask_value).all(dim=-1)  # Shape: (batch_size, seq_len)

        # Expand the mask to match the shape of predictions and targets
        mask = mask.unsqueeze(-1).expand_as(predictions)  # Shape: (batch_size, seq_len, output_dim)

        # Apply the mask to predictions and targets
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]

        # Compute the MSE loss only for non-padding tokens
        loss = F.mse_loss(masked_predictions, masked_targets, reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input_first = batch['decoder_input_first']
        decoder_input_second = batch['decoder_input_second']
        label_first = batch['label_first']
        label_second = batch['label_second']
        mask_first = batch['decoder_mask_first']
        mask_second = batch['decoder_mask_second']
        mech_type = batch['mech_type']  # New: mechanism type IDs

        # Forward pass
        pred_first, pred_second = self.forward(
            encoder_input, decoder_input_first, decoder_input_second, mask_first, mask_second, mech_type
        )

        # Compute losses for both decoders
        loss_first = self.mse_loss(pred_first, label_first)
        loss_second = self.mse_loss(pred_second, label_second)
        total_loss = loss_first + loss_second

        # Log losses
        self.log('train_loss_first', loss_first, prog_bar=True)
        self.log('train_loss_second', loss_second, prog_bar=True)
        self.log('train_loss', total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input_first = batch['decoder_input_first']
        decoder_input_second = batch['decoder_input_second']
        label_first = batch['label_first']
        label_second = batch['label_second']
        mask_first = batch['decoder_mask_first']
        mask_second = batch['decoder_mask_second']
        mech_type = batch['mech_type']  # New: mechanism type IDs

        # Forward pass
        pred_first, pred_second = self.forward(
            encoder_input, decoder_input_first, decoder_input_second, mask_first, mask_second, mech_type
        )

        # Compute losses for both decoders
        loss_first = self.mse_loss(pred_first, label_first)
        loss_second = self.mse_loss(pred_second, label_second)
        total_loss = loss_first + loss_second

        # Log losses
        self.log('val_loss_first', loss_first, sync_dist=True)
        self.log('val_loss_second', loss_second, sync_dist=True)
        self.log('val_loss', total_loss, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
