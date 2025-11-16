"""
DAY 2: SCST (Self-Critical Sequence Training) with Keyword Rewards
Goal: Improve factual accuracy and reduce hallucinations

Estimated Time: 6-8 hours on T4 x2
Expected Result: CIDEr improvement +0.1-0.2, Keyword F1 improvement +5-10%
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
from Fast_Rewards.keyword_reward import KeywordRewardFunction


def main(args):
    print("\n" + "="*80)
    print("DAY 2: SCST TRAINING (Reinforcement Learning)")
    print("="*80)
    print(f"Model: {args.model_type}")
    print(f"Warm start from: {args.pretrained_checkpoint}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Reward weights: CIDEr={args.cider_weight}, Keyword={args.keyword_weight}, Hall. Penalty={args.hallucination_penalty}")
    print("="*80 + "\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU Count: {torch.cuda.device_count()}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / "day2_scst"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")
    
    # Load data
    print("\n[1/5] Loading IU X-Ray dataset...")
    train_loader, val_loader, test_loader = get_iu_xray_dataloaders(
        data_dir=args.data_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,  # SCST needs smaller batch size
        num_workers=args.num_workers,
        max_length=args.max_length
    )
    
    # Initialize model
    print(f"\n[2/5] Initializing {args.model_type} model...")
    if args.model_type == "vit-gpt2":
        model = ViTGPT2MedicalCaptioner(
            encoder_name="google/vit-base-patch16-224-in21k",
            decoder_name="gpt2",
            use_lora=True,
            lora_r=args.lora_r,
            max_length=args.max_length,
            device=device
        )
    elif args.model_type == "blip2":
        model = Blip2MedicalCaptioner(
            model_name="Salesforce/blip2-opt-2.7b",
            use_8bit=True,
            use_lora=True,
            lora_r=args.lora_r,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load pretrained checkpoint
    if args.pretrained_checkpoint:
        print(f"\n[3/5] Loading pretrained checkpoint from {args.pretrained_checkpoint}...")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Previous best CIDEr: {checkpoint.get('best_cider', 'unknown'):.4f}")
    else:
        print("\n⚠ WARNING: No pretrained checkpoint provided!")
        print("  SCST works best when starting from a supervised baseline.")
        print("  Consider running day1_baseline.py first.\n")
    
    # Setup reward function
    print("\n[4/5] Setting up keyword-based reward function...")
    reward_fn = KeywordRewardFunction(
        cider_weight=args.cider_weight,
        keyword_weight=args.keyword_weight,
        hallucination_penalty=args.hallucination_penalty
    )
    print(f"  CIDEr weight: {args.cider_weight}")
    print(f"  Keyword F1 weight: {args.keyword_weight}")
    print(f"  Hallucination penalty: {args.hallucination_penalty}")
    
    # Setup optimizer with LOWER learning rate for RL
    print("\n[5/5] Setting up optimizer for RL...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,  # Much lower than CE training
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
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
        log_interval=20  # More frequent logging for RL
    )
    
    # Training loop
    print(f"\n[SCST Training] Starting RL training for {args.epochs} epochs...")
    print("⚠ IMPORTANT: RL training is unstable. Monitor rewards carefully!")
    print("  If rewards drop significantly, reduce learning rate or stop early.")
    print("-" * 80)
    
    best_cider = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # SCST training
        scst_stats = trainer.train_epoch_scst(
            epoch=epoch,
            reward_fn=reward_fn,
            baseline_type="greedy"
        )
        
        print(f"  Avg Reward: {scst_stats['avg_reward']:.4f}")
        print(f"  Avg Baseline: {scst_stats['avg_baseline']:.4f}")
        print(f"  Improvement: {scst_stats['avg_reward'] - scst_stats['avg_baseline']:.4f}")
        
        # Validate every N epochs
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("  Evaluating on validation set...")
            val_scores = trainer.evaluate()
            
            print("  Validation Scores:")
            for metric, score in val_scores.items():
                print(f"    {metric}: {score:.4f}")
            
            # Update history
            trainer.history["val_cider"].append(val_scores["CIDEr"])
            trainer.history["val_bleu4"].append(val_scores["BLEU-4"])
            trainer.history["epoch"] = epoch
            
            # Save best model
            is_best = val_scores["CIDEr"] > best_cider
            if is_best:
                best_cider = val_scores["CIDEr"]
                print(f"  ✓ New best CIDEr: {best_cider:.4f}")
            
            trainer.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping if RL diverges
            if scst_stats['avg_reward'] < -1.0:
                print("\n⚠ WARNING: Rewards are very negative. RL may be diverging.")
                print("  Consider stopping and using a lower learning rate.")
                if not args.no_early_stop:
                    print("  Stopping training early...")
                    break
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Learning rate: {current_lr:.2e}")
    
    print("\n" + "="*80)
    print("DAY 2 SCST TRAINING COMPLETE!")
    print("="*80)
    print(f"Best CIDEr: {best_cider:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print("\nNext steps:")
    print("  1. Compare with Day 1 baseline results")
    print("  2. If RL helped, proceed to Day 3 (ensemble)")
    print("  3. If RL hurt, use Day 1 checkpoint for final evaluation")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 2: SCST Training")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/IU_XRAY",
                        help="Path to IU X-Ray dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--pretrained_checkpoint", type=str,
                        default="checkpoints/day1_baseline/best_model.pt",
                        help="Pretrained checkpoint from Day 1")
    
    # Model
    parser.add_argument("--model_type", type=str, default="vit-gpt2",
                        choices=["vit-gpt2", "blip2"],
                        help="Model architecture")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Reward function
    parser.add_argument("--cider_weight", type=float, default=1.0,
                        help="Weight for CIDEr score")
    parser.add_argument("--keyword_weight", type=float, default=0.5,
                        help="Weight for keyword F1")
    parser.add_argument("--hallucination_penalty", type=float, default=0.3,
                        help="Penalty for hallucinated keywords")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of SCST epochs (fewer than CE)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (smaller for RL)")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate (much lower for RL)")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable early stopping on divergence")
    
    # System
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--use_multi_gpu", action="store_true", default=True,
                        help="Use multiple GPUs if available")
    
    args = parser.parse_args()
    
    main(args)
