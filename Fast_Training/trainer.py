"""
Fast Trainer for Medical Image Captioning
Optimized for Kaggle T4 x2 GPUs with 4-day constraint

Features:
- Mixed precision training (fp16)
- DataParallel for 2 GPUs
- Both Cross-Entropy and SCST training
- Fast checkpointing and evaluation
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# For evaluation
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class FastTrainer:
    """
    Fast trainer for medical image captioning
    
    Supports:
    - Phase 1: Cross-Entropy (CE) training
    - Phase 2: Self-Critical Sequence Training (SCST)
    - Mixed precision (fp16)
    - Multi-GPU (DataParallel)
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        use_fp16: bool = True,
        use_multi_gpu: bool = False,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 50
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_interval = log_interval
        
        # Setup model
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_fp16 else None
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_cider": [],
            "val_bleu4": [],
            "epoch": 0
        }
        
        # Best model tracking
        self.best_cider = 0.0
        self.best_epoch = 0
    
    def train_epoch_ce(self, epoch: int) -> float:
        """
        Train one epoch with Cross-Entropy loss
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [CE]",
            total=num_batches
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_fp16:
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels
                    )
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                    
                    # Handle DataParallel
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Free memory
            del pixel_values, labels, outputs, loss
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_epoch_scst(
        self,
        epoch: int,
        reward_fn,
        baseline_type: str = "greedy"  # "greedy" or "sample"
    ) -> Dict[str, float]:
        """
        Train one epoch with SCST (Self-Critical Sequence Training)
        
        Args:
            epoch: Current epoch number
            reward_fn: Reward function
            baseline_type: Type of baseline ("greedy" or "sample")
            
        Returns:
            Dict with training stats
        """
        self.model.train()
        total_reward = 0.0
        total_baseline = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [SCST]",
            total=num_batches
        )
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(self.device)
            gt_captions = batch["caption"]  # List of strings
            
            # Get base model (unwrap DataParallel if needed)
            base_model = self.model.module if hasattr(self.model, "module") else self.model
            
            # Sample captions with log probabilities
            with autocast() if self.use_fp16 else torch.no_grad():
                sample_ids, sample_logprobs = base_model.generate_with_logprobs(
                    pixel_values=pixel_values,
                    max_length=base_model.max_length if hasattr(base_model, "max_length") else 64,
                    temperature=1.0,
                    top_p=0.9
                )
            
            # Decode sampled captions
            if hasattr(base_model, "processor"):
                # BLIP-2
                sample_captions = base_model.processor.batch_decode(
                    sample_ids, skip_special_tokens=True
                )
            else:
                # ViT-GPT2
                sample_captions = base_model.tokenizer.batch_decode(
                    sample_ids, skip_special_tokens=True
                )
            
            # Generate baseline captions (greedy)
            with torch.no_grad():
                baseline_captions = base_model.generate(
                    pixel_values=pixel_values,
                    num_beams=1,
                    do_sample=False
                )
            
            # Compute rewards
            sample_rewards = reward_fn(
                predictions=sample_captions,
                references=[[gt] for gt in gt_captions]
            ).to(self.device)
            
            baseline_rewards = reward_fn(
                predictions=baseline_captions,
                references=[[gt] for gt in gt_captions]
            ).to(self.device)
            
            # Advantage = sample_reward - baseline_reward
            advantages = sample_rewards - baseline_rewards
            
            # Normalize advantages (stabilizes training)
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # SCST loss: -log_prob * advantage
            # Expand advantages to match sequence length
            advantages_expanded = advantages.unsqueeze(1).expand_as(sample_logprobs)
            
            # Compute loss (negative expected reward)
            loss = -(sample_logprobs * advantages_expanded).mean()
            
            # Backward pass
            if self.use_fp16:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (important for RL)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update stats
            total_reward += sample_rewards.mean().item()
            total_baseline += baseline_rewards.mean().item()
            
            pbar.set_postfix({
                "reward": f"{sample_rewards.mean().item():.3f}",
                "baseline": f"{baseline_rewards.mean().item():.3f}",
                "adv": f"{advantages.mean().item():.3f}"
            })
            
            # Free memory
            del pixel_values, sample_ids, sample_logprobs, loss
            del sample_rewards, baseline_rewards, advantages
            torch.cuda.empty_cache()
        
        stats = {
            "avg_reward": total_reward / num_batches,
            "avg_baseline": total_baseline / num_batches
        }
        
        return stats
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set
        
        Returns:
            Dict with BLEU, METEOR, ROUGE, CIDEr scores
        """
        self.model.eval()
        
        # Collect predictions and references
        predictions = {}
        references = {}
        
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        
        pbar = tqdm(self.val_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(self.device)
            gt_captions = batch["caption"]
            
            # Generate captions
            generated_captions = base_model.generate(
                pixel_values=pixel_values,
                max_length=base_model.max_length if hasattr(base_model, "max_length") else 64,
                num_beams=4,
                do_sample=False
            )
            
            # Store for metric computation
            for i, (pred, gt) in enumerate(zip(generated_captions, gt_captions)):
                idx = batch_idx * self.val_loader.batch_size + i
                predictions[idx] = [pred]
                references[idx] = [gt]
        
        # Compute metrics
        scorers = {
            "BLEU": Bleu(4),
            "METEOR": Meteor(),
            "ROUGE": Rouge(),
            "CIDEr": Cider()
        }
        
        scores = {}
        for name, scorer in scorers.items():
            score, _ = scorer.compute_score(references, predictions)
            if isinstance(score, list):
                # BLEU returns list of scores for n-grams
                for i, s in enumerate(score, 1):
                    scores[f"{name}-{i}"] = s
            else:
                scores[name] = score
        
        return scores
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "best_cider": self.best_cider
        }
        
        # Save latest checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.best_cider = checkpoint["best_cider"]
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}")


def test_fast_trainer():
    """Test fast trainer with dummy data"""
    print("\n" + "="*80)
    print("Testing Fast Trainer")
    print("="*80)
    
    # This is a placeholder - actual testing requires a real model and data
    print("Fast trainer implementation complete!")
    print("See 4Day_Scripts for usage examples")
    print("="*80)


if __name__ == "__main__":
    test_fast_trainer()
