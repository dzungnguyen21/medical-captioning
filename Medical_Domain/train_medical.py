"""
Training Script for Medical Domain (MIMIC-CXR)
Two-stage training: Supervised (XE) -> RL (SCST with RadGraph)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Shared_Modules.region_encoder import MedicalRegionExtractor
from Shared_Modules.transformer_decoder import RegionAwareTransformerDecoder
from Shared_Modules.trainer import CaptioningModel, SupervisedTrainer, RLTrainer
from Shared_Modules.hallucination_detector import MedicalHallucinationDetector
from Shared_Modules.reward_functions import MedicalReward
from Shared_Modules.metrics import ComprehensiveEvaluator
from Medical_Domain.data_loader import get_medical_dataloader


def train_supervised(args, model, train_loader, val_loader, vocab):
    """Phase 1: Supervised training with Cross-Entropy loss."""
    
    print("=" * 80)
    print("PHASE 1: Supervised Training (Cross-Entropy Loss)")
    print("=" * 80)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr_xe)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs_xe):
        print(f"\n--- Epoch {epoch+1}/{args.epochs_xe} ---")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'vocab': vocab
            }
            save_path = os.path.join(args.checkpoint_dir, 'best_xe_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'vocab': vocab
            }
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
    
    print("\n" + "=" * 80)
    print("Phase 1 completed!")
    print("=" * 80 + "\n")


def train_rl(args, model, train_loader, val_loader, vocab, medical_detector):
    """Phase 2: Reinforcement Learning with SCST and RadGraph."""
    
    print("=" * 80)
    print("PHASE 2: Reinforcement Learning (SCST with RadGraph F1)")
    print("=" * 80)
    
    # Load best XE model
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_xe_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded XE model from {checkpoint_path}")
    else:
        print("Warning: No XE model found, starting RL from scratch")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr_rl)
    
    # Create reward function
    idx_to_word = {v: k for k, v in vocab.items()}
    reward_function = MedicalReward(
        idx_to_word=idx_to_word,
        medical_detector=medical_detector,
        alpha=args.reward_cider_weight,
        beta=args.reward_radgraph_weight
    )
    
    trainer = RLTrainer(
        model=model,
        optimizer=optimizer,
        reward_function=reward_function,
        device=args.device,
        grad_clip=args.grad_clip,
        baseline_type='greedy',
        entropy_weight=args.entropy_weight
    )
    
    best_reward = -float('inf')
    
    for epoch in range(args.epochs_rl):
        print(f"\n--- Epoch {epoch+1}/{args.epochs_rl} ---")
        
        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch,
            start_token_id=vocab['<START>'],
            end_token_id=vocab['<END>'],
            max_len=args.max_seq_len
        )
        
        print(f"RL Loss: {train_metrics['loss']:.4f}")
        print(f"Reward: {train_metrics['reward']:.4f}")
        print(f"Baseline: {train_metrics['baseline_reward']:.4f}")
        print(f"Advantage: {train_metrics['advantage']:.4f}")
        
        # Save checkpoint
        if train_metrics['reward'] > best_reward:
            best_reward = train_metrics['reward']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': train_metrics['reward'],
                'vocab': vocab
            }
            save_path = os.path.join(args.checkpoint_dir, 'best_rl_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best RL model to {save_path}")
    
    print("\n" + "=" * 80)
    print("Phase 2 completed!")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Medical Domain Captioning Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data/MIMIC-CXR',
                        help='Path to MIMIC-CXR dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/medical',
                        help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--region_feature_dim', type=int, default=2048,
                        help='Region feature dimension')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward dimension')
    parser.add_argument('--num_regions', type=int, default=36,
                        help='Number of regions to extract')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='Maximum sequence length (longer for medical reports)')
    
    # Training parameters - Phase 1 (Supervised)
    parser.add_argument('--epochs_xe', type=int, default=50,
                        help='Number of epochs for XE training')
    parser.add_argument('--lr_xe', type=float, default=5e-4,
                        help='Learning rate for XE training')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    
    # Training parameters - Phase 2 (RL)
    parser.add_argument('--epochs_rl', type=int, default=30,
                        help='Number of epochs for RL training')
    parser.add_argument('--lr_rl', type=float, default=5e-6,
                        help='Learning rate for RL training')
    parser.add_argument('--reward_cider_weight', type=float, default=1.0,
                        help='Weight for CIDEr reward')
    parser.add_argument('--reward_radgraph_weight', type=float, default=2.0,
                        help='Weight for RadGraph F1 reward')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Entropy regularization weight')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (smaller for medical images)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping value')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip_xe', action='store_true',
                        help='Skip supervised training')
    parser.add_argument('--skip_rl', action='store_true',
                        help='Skip RL training')
    parser.add_argument('--use_radgraph', action='store_true',
                        help='Use RadGraph for reward (requires installation)')
    parser.add_argument('--use_chexbert', action='store_true',
                        help='Use CheXbert for evaluation')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    
    # Build vocabulary from training set
    temp_loader = get_medical_dataloader(
        dataset_name='mimic_cxr',
        data_dir=args.data_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        vocab=None  # Will build vocab
    )
    vocab = temp_loader.dataset.vocab
    
    # Save vocabulary
    vocab_path = os.path.join(args.checkpoint_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create data loaders with vocabulary
    train_loader = get_medical_dataloader(
        dataset_name='mimic_cxr',
        data_dir=args.data_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        vocab=vocab
    )
    
    val_loader = get_medical_dataloader(
        dataset_name='mimic_cxr',
        data_dir=args.data_dir,
        split='validate',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        vocab=vocab
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nBuilding model...")
    
    region_encoder = MedicalRegionExtractor(
        pretrained=True,
        num_regions=args.num_regions,
        feature_dim=args.region_feature_dim,
        device=args.device
    ).to(args.device)
    
    decoder = RegionAwareTransformerDecoder(
        vocab_size=len(vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=0.1,
        pad_token_id=vocab['<PAD>']
    ).to(args.device)
    
    model = CaptioningModel(
        region_encoder=region_encoder,
        decoder=decoder,
        region_feature_dim=args.region_feature_dim,
        decoder_dim=args.d_model
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create medical hallucination detector
    medical_detector = MedicalHallucinationDetector(
        vocab=vocab,
        use_chexbert=args.use_chexbert,
        use_radgraph=args.use_radgraph
    )
    
    # Train Phase 1: Supervised
    if not args.skip_xe:
        train_supervised(args, model, train_loader, val_loader, vocab)
    
    # Train Phase 2: RL
    if not args.skip_rl:
        train_rl(args, model, train_loader, val_loader, vocab, medical_detector)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
