"""
Training Module for Image Captioning
Includes both Supervised (XE) and Reinforcement Learning (SCST) training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class CaptioningModel(nn.Module):
    """
    Complete captioning model combining encoder and decoder.
    """
    
    def __init__(
        self,
        region_encoder,
        decoder,
        region_feature_dim: int = 2048,
        decoder_dim: int = 512
    ):
        super().__init__()
        
        self.region_encoder = region_encoder
        self.decoder = decoder
        
        # Project region features to decoder dimension
        self.feature_adapter = nn.Linear(region_feature_dim, decoder_dim)
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass for training.
        
        Args:
            images: [batch_size, 3, H, W]
            captions: [batch_size, seq_len] target captions
            return_attention: Whether to return attention weights
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            attention_weights: Optional attention weights
        """
        # Extract region features
        encoder_output = self.region_encoder(images, return_detections=True)
        region_features = encoder_output['region_features']  # [B, num_regions, feature_dim]
        
        # Adapt features to decoder dimension
        region_features = self.feature_adapter(region_features)
        
        # Decode
        logits, attention_weights = self.decoder(
            captions, region_features, return_attention=return_attention
        )
        
        return logits, attention_weights, encoder_output
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int = 50,
        beam_size: int = 1,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Generate captions for images.
        """
        # Extract region features
        encoder_output = self.region_encoder(images, return_detections=True)
        region_features = encoder_output['region_features']
        
        # Adapt features
        region_features = self.feature_adapter(region_features)
        
        # Generate
        generated, attention = self.decoder.generate(
            region_features,
            start_token_id,
            end_token_id,
            max_len=max_len,
            beam_size=beam_size,
            return_attention=return_attention
        )
        
        return generated, attention, encoder_output


class SupervisedTrainer:
    """
    Supervised trainer using Cross-Entropy loss (Teacher Forcing).
    This is Phase 1 of training.
    """
    
    def __init__(
        self,
        model: CaptioningModel,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        grad_clip: float = 5.0,
        label_smoothing: float = 0.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        
        # Cross-entropy loss with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignore padding
            label_smoothing=label_smoothing
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            captions = batch['captions'].to(self.device)  # [B, seq_len]
            
            # Forward pass
            # Input: captions without last token (teacher forcing)
            # Target: captions without first token (shifted)
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            
            logits, _, _ = self.model(images, input_captions)
            
            # Compute loss
            # Reshape for cross-entropy: [B*seq_len, vocab_size] and [B*seq_len]
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.reshape(-1, vocab_size)
            targets = target_captions.reshape(-1)
            
            loss = self.criterion(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            captions = batch['captions'].to(self.device)
            
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            
            logits, _, _ = self.model(images, input_captions)
            
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.reshape(-1, vocab_size)
            targets = target_captions.reshape(-1)
            
            loss = self.criterion(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}


class RLTrainer:
    """
    Reinforcement Learning trainer using Self-Critical Sequence Training (SCST).
    This is Phase 2 of training.
    """
    
    def __init__(
        self,
        model: CaptioningModel,
        optimizer: torch.optim.Optimizer,
        reward_function,
        device: str = 'cuda',
        grad_clip: float = 5.0,
        baseline_type: str = 'greedy',  # 'greedy' or 'average'
        entropy_weight: float = 0.0
    ):
        """
        Args:
            model: Captioning model
            optimizer: Optimizer
            reward_function: Function that computes rewards given captions
            device: Device to use
            grad_clip: Gradient clipping value
            baseline_type: Type of baseline ('greedy' or 'average')
            entropy_weight: Weight for entropy regularization
        """
        self.model = model
        self.optimizer = optimizer
        self.reward_function = reward_function
        self.device = device
        self.grad_clip = grad_clip
        self.baseline_type = baseline_type
        self.entropy_weight = entropy_weight
    
    def compute_sample_logprobs(
        self,
        logits: torch.Tensor,
        sampled_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Compute log probabilities of sampled sequences.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            sampled_ids: [batch_size, seq_len]
            pad_token_id: Padding token ID
            
        Returns:
            log_probs: [batch_size] sum of log probs for each sequence
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]
        
        # Gather log probs of sampled tokens
        sampled_log_probs = log_probs.gather(
            dim=-1, 
            index=sampled_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, seq_len]
        
        # Mask out padding tokens
        mask = (sampled_ids != pad_token_id).float()
        sampled_log_probs = sampled_log_probs * mask
        
        # Sum log probs for each sequence
        sequence_log_probs = sampled_log_probs.sum(dim=-1)  # [B]
        
        return sequence_log_probs
    
    def sample_captions(
        self,
        images: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int = 50,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample captions using the model.
        
        Returns:
            sampled_ids: [batch_size, seq_len]
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size = images.size(0)
        device = images.device
        
        # Extract region features
        encoder_output = self.model.region_encoder(images, return_detections=True)
        region_features = encoder_output['region_features']
        region_features = self.model.feature_adapter(region_features)
        
        # Initialize
        sampled_ids = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long, device=device
        )
        all_logits = []
        
        for step in range(max_len):
            # Forward pass
            logits, _ = self.model.decoder(sampled_ids, region_features)
            
            # Get logits for last token
            current_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            all_logits.append(current_logits.unsqueeze(1))
            
            # Sample next token
            probs = F.softmax(current_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Append to sequence
            sampled_ids = torch.cat([sampled_ids, next_token], dim=1)
            
            # Check if all sequences have generated end token
            if (next_token.squeeze(-1) == end_token_id).all():
                break
        
        # Stack logits
        all_logits = torch.cat(all_logits, dim=1)  # [B, seq_len, vocab_size]
        
        # Remove start token from sampled_ids
        sampled_ids = sampled_ids[:, 1:]  # [B, seq_len]
        
        return sampled_ids, all_logits, encoder_output
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        start_token_id: int,
        end_token_id: int,
        max_len: int = 50
    ) -> Dict[str, float]:
        """
        Train for one epoch using SCST.
        """
        self.model.train()
        
        total_loss = 0.0
        total_reward = 0.0
        total_baseline_reward = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [RL Train]')
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            reference_captions = batch['captions_text']  # List of caption strings
            
            # Sample captions
            sampled_ids, sampled_logits, encoder_output = self.sample_captions(
                images, start_token_id, end_token_id, max_len
            )
            
            # Generate baseline captions (greedy)
            with torch.no_grad():
                if self.baseline_type == 'greedy':
                    baseline_ids, _, _ = self.model.generate(
                        images, start_token_id, end_token_id, 
                        max_len=max_len, beam_size=1
                    )
                else:
                    # For 'average', use sampled captions as baseline (REINFORCE)
                    baseline_ids = sampled_ids
            
            # Compute rewards for sampled captions
            sample_rewards = self.reward_function(
                sampled_ids, reference_captions, encoder_output
            )
            
            # Compute rewards for baseline captions
            with torch.no_grad():
                baseline_rewards = self.reward_function(
                    baseline_ids, reference_captions, encoder_output
                )
            
            # Compute advantage
            advantages = sample_rewards - baseline_rewards  # [B]
            
            # Compute log probabilities
            log_probs = self.compute_sample_logprobs(
                sampled_logits, sampled_ids
            )  # [B]
            
            # RL loss: -E[advantage * log_prob]
            rl_loss = -(advantages * log_probs).mean()
            
            # Optional: Add entropy regularization to encourage exploration
            if self.entropy_weight > 0:
                probs = F.softmax(sampled_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                rl_loss = rl_loss - self.entropy_weight * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            rl_loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += rl_loss.item()
            total_reward += sample_rewards.mean().item()
            total_baseline_reward += baseline_rewards.mean().item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': rl_loss.item(),
                'reward': sample_rewards.mean().item(),
                'baseline': baseline_rewards.mean().item()
            })
        
        return {
            'loss': total_loss / num_batches,
            'reward': total_reward / num_batches,
            'baseline_reward': total_baseline_reward / num_batches,
            'advantage': (total_reward - total_baseline_reward) / num_batches
        }
