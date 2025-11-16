"""
DAY 3: Ensemble and Fine-tuning
Goal: Combine multiple models or checkpoints for better performance

Options:
1. Ensemble different epochs from SCST
2. Ensemble baseline + SCST models
3. Further fine-tuning with hybrid loss

Estimated Time: 2-4 hours
Expected Result: Additional +0.02-0.05 CIDEr improvement
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from Fast_Models.vit_gpt2_wrapper import ViTGPT2MedicalCaptioner
from Fast_Models.blip2_wrapper import Blip2MedicalCaptioner
from Fast_Data.iu_xray_loader import get_iu_xray_dataloaders

# For evaluation
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def load_model(checkpoint_path: str, model_type: str, device: str):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model
    if model_type == "vit-gpt2":
        model = ViTGPT2MedicalCaptioner(
            encoder_name="google/vit-base-patch16-224-in21k",
            decoder_name="gpt2",
            use_lora=True,
            device=device
        )
    elif model_type == "blip2":
        model = Blip2MedicalCaptioner(
            model_name="Salesforce/blip2-opt-2.7b",
            use_8bit=True,
            use_lora=True,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def ensemble_generate(
    models: List,
    pixel_values: torch.Tensor,
    max_length: int = 128,
    num_beams: int = 4,
    ensemble_method: str = "voting"
) -> List[str]:
    """
    Generate captions using ensemble of models
    
    Args:
        models: List of models
        pixel_values: Image tensor
        max_length: Maximum sequence length
        num_beams: Number of beams
        ensemble_method: "voting" or "reranking"
        
    Returns:
        List of generated captions
    """
    batch_size = pixel_values.shape[0]
    all_captions = []
    all_scores = []
    
    # Generate from each model
    for model in models:
        with torch.no_grad():
            if ensemble_method == "voting":
                # Beam search from each model
                captions = model.generate(
                    pixel_values=pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False
                )
                all_captions.append(captions)
            
            elif ensemble_method == "reranking":
                # Generate multiple candidates
                captions = model.generate(
                    pixel_values=pixel_values,
                    max_length=max_length,
                    num_beams=num_beams * 2,
                    num_return_sequences=5,  # Get top 5
                    do_sample=False
                )
                all_captions.append(captions)
    
    # Combine results
    if ensemble_method == "voting":
        # Majority voting or select most common caption
        final_captions = []
        for i in range(batch_size):
            # Get all captions for this image
            captions_for_image = [caps[i] for caps in all_captions]
            
            # Simple voting: pick most common
            from collections import Counter
            counter = Counter(captions_for_image)
            most_common = counter.most_common(1)[0][0]
            final_captions.append(most_common)
        
        return final_captions
    
    else:  # reranking
        # This is more complex - for now, just return first model's output
        return all_captions[0][:batch_size]


@torch.no_grad()
def evaluate_ensemble(
    models: List,
    data_loader,
    device: str,
    ensemble_method: str = "voting"
) -> Dict[str, float]:
    """Evaluate ensemble of models"""
    predictions = {}
    references = {}
    
    pbar = tqdm(data_loader, desc="Evaluating ensemble")
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        gt_captions = batch["caption"]
        
        # Generate with ensemble
        generated_captions = ensemble_generate(
            models=models,
            pixel_values=pixel_values,
            max_length=128,
            num_beams=4,
            ensemble_method=ensemble_method
        )
        
        # Store for metrics
        for i, (pred, gt) in enumerate(zip(generated_captions, gt_captions)):
            idx = batch_idx * data_loader.batch_size + i
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
            for i, s in enumerate(score, 1):
                scores[f"{name}-{i}"] = s
        else:
            scores[name] = score
    
    return scores


def main(args):
    print("\n" + "="*80)
    print("DAY 3: ENSEMBLE AND FINE-TUNING")
    print("="*80)
    print(f"Model type: {args.model_type}")
    print(f"Ensemble method: {args.ensemble_method}")
    print(f"Checkpoints to ensemble:")
    for ckpt in args.checkpoints:
        print(f"  - {ckpt}")
    print("="*80 + "\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Load data
    print("[1/3] Loading validation and test data...")
    _, val_loader, test_loader = get_iu_xray_dataloaders(
        data_dir=args.data_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=128
    )
    
    # Load models
    print(f"\n[2/3] Loading {len(args.checkpoints)} models for ensemble...")
    models = []
    for ckpt_path in args.checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"⚠ Warning: Checkpoint not found: {ckpt_path}")
            continue
        model = load_model(ckpt_path, args.model_type, device)
        models.append(model)
    
    if len(models) == 0:
        print("✗ No valid checkpoints found. Exiting.")
        return
    
    if len(models) == 1:
        print("⚠ Only one model loaded. This is not an ensemble.")
        print("  Will evaluate single model instead.\n")
    
    print(f"✓ Loaded {len(models)} models successfully\n")
    
    # Evaluate ensemble on validation set
    print("[3/3] Evaluating ensemble on validation set...")
    val_scores = evaluate_ensemble(
        models=models,
        data_loader=val_loader,
        device=device,
        ensemble_method=args.ensemble_method
    )
    
    print("\nValidation Results:")
    print("-" * 40)
    for metric, score in sorted(val_scores.items()):
        print(f"  {metric:12s}: {score:.4f}")
    
    # Evaluate on test set if requested
    if args.evaluate_test:
        print("\n[Bonus] Evaluating ensemble on test set...")
        test_scores = evaluate_ensemble(
            models=models,
            data_loader=test_loader,
            device=device,
            ensemble_method=args.ensemble_method
        )
        
        print("\nTest Results:")
        print("-" * 40)
        for metric, score in sorted(test_scores.items()):
            print(f"  {metric:12s}: {score:.4f}")
        
        # Save results
        import json
        results = {
            "validation": val_scores,
            "test": test_scores,
            "ensemble_method": args.ensemble_method,
            "num_models": len(models)
        }
        
        results_path = Path(args.checkpoint_dir) / "day3_ensemble_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "="*80)
    print("DAY 3 ENSEMBLE EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nBest CIDEr: {val_scores['CIDEr']:.4f}")
    print("\nNext steps:")
    print("  1. Compare ensemble vs individual models")
    print("  2. If ensemble helps, use it for Day 4 final evaluation")
    print("  3. Otherwise, use best single model")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 3: Ensemble Evaluation")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/IU_XRAY",
                        help="Path to IU X-Ray dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    
    # Model
    parser.add_argument("--model_type", type=str, default="vit-gpt2",
                        choices=["vit-gpt2", "blip2"],
                        help="Model architecture")
    
    # Ensemble
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default=[
                            "checkpoints/day1_baseline/best_model.pt",
                            "checkpoints/day2_scst/best_model.pt"
                        ],
                        help="List of checkpoint paths to ensemble")
    parser.add_argument("--ensemble_method", type=str, default="voting",
                        choices=["voting", "reranking"],
                        help="Ensemble method")
    
    # Evaluation
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--evaluate_test", action="store_true", default=True,
                        help="Also evaluate on test set")
    
    args = parser.parse_args()
    
    main(args)
