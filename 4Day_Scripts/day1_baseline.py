"""
DAY 1: Baseline Training with Cross-Entropy Loss
Goal: Get a working model with supervised learning

Estimated Time: 4-6 hours on T4 x2
Expected Result: BLEU-4 ~0.15-0.20, CIDEr ~0.3-0.5
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from Fast_Models.vit_gpt2_wrapper import ViTGPT2MedicalCaptioner
from Fast_Models.blip2_wrapper import Blip2MedicalCaptioner
from Fast_Data.iu_xray_loader import get_iu_xray_dataloaders
from Fast_Training.trainer import FastTrainer


def main(args):
    print("\n" + "="*80)
    print("DAY 1: BASELINE TRAINING (Cross-Entropy)")
    print("="*80)
    print(f"Model: {args.model_type}")
    print(f"Data: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Use fp16: {args.use_fp16}")
    print(f"Multi-GPU: {args.use_multi_gpu}")
    print("="*80 + "\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / "day1_baseline"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")
    
    # Load data
    print("\n[1/4] Loading IU X-Ray dataset...")
    train_loader, val_loader, test_loader = get_iu_xray_dataloaders(
        data_dir=args.data_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )
    
    # Initialize model
    print(f"\n[2/4] Initializing {args.model_type} model...")
    if args.model_type == "vit-gpt2":
        model = ViTGPT2MedicalCaptioner(
            encoder_name="google/vit-base-patch16-224-in21k",
            decoder_name="gpt2",
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            max_length=args.max_length,
            device=device
        )
    elif args.model_type == "blip2":
        model = Blip2MedicalCaptioner(
            model_name="Salesforce/blip2-opt-2.7b",
            use_8bit=args.use_8bit,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Setup optimizer
    print("\n[3/4] Setting up optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Learning rate scheduler (cosine)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Initialize trainer
    trainer = FastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        use_fp16=args.use_fp16,
        use_multi_gpu=args.use_multi_gpu,
        checkpoint_dir=checkpoint_dir,
        log_interval=50
    )
    
    # Training loop
    print(f"\n[4/4] Starting training for {args.epochs} epochs...")
    print("-" * 80)
    
    best_cider = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch_ce(epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate every N epochs
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("  Evaluating on validation set...")
            val_scores = trainer.evaluate()
            
            print("  Validation Scores:")
            for metric, score in val_scores.items():
                print(f"    {metric}: {score:.4f}")
            
            # Update history
            trainer.history["train_loss"].append(train_loss)
            trainer.history["val_cider"].append(val_scores["CIDEr"])
            trainer.history["val_bleu4"].append(val_scores["BLEU-4"])
            trainer.history["epoch"] = epoch
            
            # Save best model
            is_best = val_scores["CIDEr"] > best_cider
            if is_best:
                best_cider = val_scores["CIDEr"]
                print(f"  ✓ New best CIDEr: {best_cider:.4f}")
            
            trainer.save_checkpoint(epoch, is_best=is_best)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Learning rate: {current_lr:.2e}")
    
    print("\n" + "="*80)
    print("DAY 1 TRAINING COMPLETE!")
    print("="*80)
    print(f"Best CIDEr: {best_cider:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print("\nNext steps:")
    print("  1. Check results in checkpoints/day1_baseline/")
    print("  2. Run Day 2 script for SCST training")
    print("  3. Expected improvement: CIDEr +0.1-0.2 after RL")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 1: Baseline Training")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/IU_XRAY",
                        help="Path to IU X-Ray dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    # Model
    parser.add_argument("--model_type", type=str, default="vit-gpt2",
                        choices=["vit-gpt2", "blip2"],
                        help="Model architecture")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--use_8bit", action="store_true", default=True,
                        help="Use 8-bit quantization (BLIP-2 only)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=2,
                        help="Evaluate every N epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    # System
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--use_multi_gpu", action="store_true", default=True,
                        help="Use multiple GPUs if available")
    
    args = parser.parse_args()
    
    main(args)
