"""
Transformer Decoder with Cross-Attention for Region-Aware Captioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, 1, seq_len_q, seq_len_k] or [batch_size, seq_len_q, seq_len_k]
            return_attention: Whether to return attention weights
            
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k] (optional)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # [B, seq_len_q, d_model]
        K = self.W_k(key)    # [B, seq_len_k, d_model]
        V = self.W_v(value)  # [B, seq_len_v, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, seq_len_q, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, seq_len_k, d_k]
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, seq_len_v, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, seq_len_q, seq_len_k]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # [B, num_heads, seq_len_q, seq_len_k]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [B, num_heads, seq_len_q, d_k]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()  # [B, seq_len_q, num_heads, d_k]
        output = output.view(batch_size, -1, self.d_model)  # [B, seq_len_q, d_model]
        
        # Final linear projection
        output = self.W_o(output)
        
        if return_attention:
            return output, attention_weights
        return output, None


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerDecoderLayer(nn.Module):
    """Single layer of Transformer decoder with cross-attention."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention (for attending to image regions)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Target sequence [batch_size, tgt_len, d_model]
            memory: Encoder output (region features) [batch_size, num_regions, d_model]
            tgt_mask: Mask for target sequence (causal mask)
            memory_mask: Mask for encoder output
            return_attention: Whether to return cross-attention weights
            
        Returns:
            output: [batch_size, tgt_len, d_model]
            cross_attention: [batch_size, num_heads, tgt_len, num_regions] (optional)
        """
        # Self-attention with residual connection
        self_attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        
        # Cross-attention with residual connection
        cross_attn_out, cross_attention = self.cross_attn(
            x, memory, memory, 
            mask=memory_mask,
            return_attention=return_attention
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        
        return x, cross_attention


class RegionAwareTransformerDecoder(nn.Module):
    """
    Transformer decoder that attends to image regions.
    Used for both General and Medical image captioning.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Returns:
            mask: [seq_len, seq_len] where mask[i, j] = 0 if j > i (future tokens)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask.
        
        Args:
            seq: [batch_size, seq_len]
            
        Returns:
            mask: [batch_size, 1, seq_len] where mask[i, j] = 0 if seq[i, j] is padding
        """
        mask = (seq != self.pad_token_id).unsqueeze(1)  # [B, 1, seq_len]
        return mask
    
    def forward(
        self,
        tgt: torch.Tensor,
        region_features: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.
        
        Args:
            tgt: Target sequence [batch_size, tgt_len] (token IDs)
            region_features: Region features from encoder [batch_size, num_regions, d_model]
            tgt_mask: Target mask (usually causal mask)
            memory_mask: Memory mask
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            logits: [batch_size, tgt_len, vocab_size]
            attention_weights: List of attention weights from each layer (optional)
        """
        batch_size, tgt_len = tgt.size()
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt_len, tgt.device)
            tgt_mask = tgt_mask.unsqueeze(0)  # [1, tgt_len, tgt_len]
        
        # Token embedding + positional encoding
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)  # [B, tgt_len, d_model]
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        attention_weights_list = []
        for layer in self.layers:
            x, attn = layer(
                x, region_features, 
                tgt_mask=tgt_mask, 
                memory_mask=memory_mask,
                return_attention=return_attention
            )
            if return_attention and attn is not None:
                attention_weights_list.append(attn)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # [B, tgt_len, vocab_size]
        
        if return_attention:
            return logits, attention_weights_list
        return logits, None
    
    @torch.no_grad()
    def generate(
        self,
        region_features: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        beam_size: int = 1,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate captions autoregressively.
        
        Args:
            region_features: [batch_size, num_regions, d_model]
            start_token_id: Start token ID
            end_token_id: End token ID
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            beam_size: Beam search size (1 = greedy)
            return_attention: Whether to return attention weights
            
        Returns:
            generated_ids: [batch_size, seq_len]
            attention_maps: Attention weights over regions (optional)
        """
        if beam_size > 1:
            return self._beam_search(
                region_features, start_token_id, end_token_id, 
                max_len, beam_size, return_attention
            )
        else:
            return self._greedy_search(
                region_features, start_token_id, end_token_id,
                max_len, temperature, top_k, top_p, return_attention
            )
    
    def _greedy_search(
        self,
        region_features: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Greedy decoding or sampling."""
        batch_size = region_features.size(0)
        device = region_features.device
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        attention_maps = [] if return_attention else None
        
        for _ in range(max_len):
            # Forward pass
            logits, attn_weights = self.forward(
                generated, region_features, return_attention=return_attention
            )
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Store attention if requested
            if return_attention and attn_weights:
                # Average attention across layers and heads
                avg_attn = torch.stack(attn_weights).mean(dim=0).mean(dim=1)  # [B, tgt_len, num_regions]
                attention_maps.append(avg_attn[:, -1, :])  # Last token's attention
            
            # Check if all sequences have generated end token
            if (next_token == end_token_id).all():
                break
        
        if return_attention and attention_maps:
            attention_maps = torch.stack(attention_maps, dim=1)  # [B, gen_len, num_regions]
        
        return generated, attention_maps
    
    def _beam_search(
        self,
        region_features: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int,
        beam_size: int,
        return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Beam search decoding."""
        batch_size = region_features.size(0)
        device = region_features.device
        
        # Expand region features for beam search
        region_features = region_features.unsqueeze(1).repeat(1, beam_size, 1, 1)
        region_features = region_features.view(batch_size * beam_size, -1, self.d_model)
        
        # Initialize beams
        sequences = torch.full(
            (batch_size, beam_size, 1), start_token_id, dtype=torch.long, device=device
        )
        scores = torch.zeros(batch_size, beam_size, device=device)
        scores[:, 1:] = -float('Inf')  # Only use first beam initially
        
        for step in range(max_len):
            # Flatten for batch processing
            current_sequences = sequences.view(batch_size * beam_size, -1)
            
            # Forward pass
            logits, _ = self.forward(current_sequences, region_features)
            logits = logits[:, -1, :]  # [batch_size * beam_size, vocab_size]
            
            # Reshape logits
            logits = logits.view(batch_size, beam_size, -1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute scores for all possible next tokens
            next_scores = scores.unsqueeze(-1) + log_probs  # [B, beam_size, vocab_size]
            next_scores = next_scores.view(batch_size, -1)  # [B, beam_size * vocab_size]
            
            # Get top beam_size candidates
            top_scores, top_indices = next_scores.topk(beam_size, dim=-1)
            
            # Determine which beam and token each top candidate corresponds to
            beam_indices = top_indices // self.vocab_size
            token_indices = top_indices % self.vocab_size
            
            # Update sequences
            new_sequences = []
            for b in range(batch_size):
                batch_sequences = []
                for k in range(beam_size):
                    beam_idx = beam_indices[b, k]
                    token_idx = token_indices[b, k]
                    
                    prev_seq = sequences[b, beam_idx]
                    new_seq = torch.cat([prev_seq, token_idx.unsqueeze(0)])
                    batch_sequences.append(new_seq)
                
                new_sequences.append(torch.stack(batch_sequences))
            
            sequences = torch.stack(new_sequences)
            scores = top_scores
            
            # Check if all beams have generated end token
            all_ended = (sequences[:, :, -1] == end_token_id).all()
            if all_ended:
                break
        
        # Return best sequence from each batch
        best_sequences = sequences[:, 0, :]  # [batch_size, seq_len]
        
        # Note: Attention weights for beam search would require storing all beam attention
        # For simplicity, returning None here
        return best_sequences, None
