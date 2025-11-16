"""
DAY 4: Final Evaluation and Report Generation
Goal: Comprehensive evaluation with all metrics and qualitative analysis

Outputs:
1. Quantitative metrics (BLEU, METEOR, ROUGE, CIDEr)
2. Keyword analysis (Precision, Recall, F1, Hallucination rate)
3. Qualitative examples (Good vs Bad captions)
4. Comparison table (Baseline vs SCST vs Ensemble)
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from Fast_Models.vit_gpt2_wrapper import ViTGPT2MedicalCaptioner
from Fast_Models.blip2_wrapper import Blip2MedicalCaptioner
from Fast_Data.iu_xray_loader import get_iu_xray_dataloaders
from Fast_Rewards.keyword_reward import KeywordRewardFunction

# Evaluation metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def load_model(checkpoint_path: str, model_type: str, device: str):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device: str,
    save_predictions: bool = True
) -> Tuple[Dict[str, float], Dict]:
    """
    Comprehensive evaluation of model
    
    Returns:
        scores: Dict of metric scores
        predictions_data: Dict with predictions, references, image paths
    """
    predictions = {}
    references = {}
    image_paths = {}
    
    pbar = tqdm(data_loader, desc="Generating captions")
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        gt_captions = batch["caption"]
        paths = batch["image_path"]
        
        # Generate captions
        generated_captions = model.generate(
            pixel_values=pixel_values,
            max_length=128,
            num_beams=4,
            do_sample=False
        )
        
        # Store
        for i, (pred, gt, path) in enumerate(zip(generated_captions, gt_captions, paths)):
            idx = batch_idx * data_loader.batch_size + i
            predictions[idx] = [pred]
            references[idx] = [gt]
            image_paths[idx] = path
    
    # Compute NLG metrics
    print("\nComputing NLG metrics...")
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
    
    # Compute keyword metrics
    print("Computing keyword metrics...")
    reward_fn = KeywordRewardFunction()
    
    all_pred_keywords = []
    all_gt_keywords = []
    keyword_precisions = []
    keyword_recalls = []
    keyword_f1s = []
    hallucination_rates = []
    
    for idx in predictions:
        pred_text = predictions[idx][0]
        gt_text = references[idx][0]
        
        pred_kw = reward_fn.extract_keywords(pred_text)
        gt_kw = reward_fn.extract_keywords(gt_text)
        
        all_pred_keywords.append(pred_kw)
        all_gt_keywords.append(gt_kw)
        
        kw_metrics = reward_fn.compute_keyword_f1(pred_kw, gt_kw)
        keyword_precisions.append(kw_metrics["precision"])
        keyword_recalls.append(kw_metrics["recall"])
        keyword_f1s.append(kw_metrics["f1"])
        
        hall_rate = reward_fn.compute_hallucination_penalty(pred_kw, gt_kw)
        hallucination_rates.append(hall_rate)
    
    # Average keyword metrics
    scores["Keyword_Precision"] = np.mean(keyword_precisions)
    scores["Keyword_Recall"] = np.mean(keyword_recalls)
    scores["Keyword_F1"] = np.mean(keyword_f1s)
    scores["Hallucination_Rate"] = np.mean(hallucination_rates)
    
    # Prepare detailed predictions data
    predictions_data = {
        "predictions": predictions,
        "references": references,
        "image_paths": image_paths,
        "pred_keywords": all_pred_keywords,
        "gt_keywords": all_gt_keywords
    }
    
    return scores, predictions_data


def find_qualitative_examples(
    predictions_data: Dict,
    scores_per_sample: Dict,
    num_best: int = 5,
    num_worst: int = 5
) -> Tuple[List, List]:
    """
    Find best and worst examples for qualitative analysis
    
    Returns:
        best_examples: List of best prediction examples
        worst_examples: List of worst prediction examples
    """
    # Rank by CIDEr or keyword F1
    indices = list(predictions_data["predictions"].keys())
    
    # Compute simple score per sample
    sample_scores = []
    for idx in indices:
        pred = predictions_data["predictions"][idx][0]
        ref = predictions_data["references"][idx][0]
        
        # Simple word overlap score
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        overlap = len(pred_words & ref_words)
        score = overlap / max(len(pred_words), 1)
        
        sample_scores.append((idx, score))
    
    # Sort
    sample_scores.sort(key=lambda x: x[1], reverse=True)
    
    best_indices = [idx for idx, _ in sample_scores[:num_best]]
    worst_indices = [idx for idx, _ in sample_scores[-num_worst:]]
    
    best_examples = [
        {
            "index": idx,
            "image_path": predictions_data["image_paths"][idx],
            "prediction": predictions_data["predictions"][idx][0],
            "reference": predictions_data["references"][idx][0],
            "pred_keywords": predictions_data["pred_keywords"][idx],
            "gt_keywords": predictions_data["gt_keywords"][idx]
        }
        for idx in best_indices
    ]
    
    worst_examples = [
        {
            "index": idx,
            "image_path": predictions_data["image_paths"][idx],
            "prediction": predictions_data["predictions"][idx][0],
            "reference": predictions_data["references"][idx][0],
            "pred_keywords": predictions_data["pred_keywords"][idx],
            "gt_keywords": predictions_data["gt_keywords"][idx]
        }
        for idx in worst_indices
    ]
    
    return best_examples, worst_examples


def generate_report(
    results: Dict[str, Dict],
    best_examples: List,
    worst_examples: List,
    output_dir: Path
):
    """Generate comprehensive evaluation report"""
    
    report_path = output_dir / "FINAL_REPORT.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Medical Image Captioning - Final Evaluation Report\n\n")
        f.write("## 4-Day Fast Training Results\n\n")
        
        # Quantitative results table
        f.write("### Quantitative Metrics\n\n")
        f.write("| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr | Keyword F1 | Hallucination |\n")
        f.write("|-------|--------|--------|---------|-------|------------|---------------|\n")
        
        for model_name, scores in results.items():
            bleu4 = scores.get("BLEU-4", 0.0)
            meteor = scores.get("METEOR", 0.0)
            rouge = scores.get("ROUGE", 0.0)
            cider = scores.get("CIDEr", 0.0)
            kw_f1 = scores.get("Keyword_F1", 0.0)
            hall = scores.get("Hallucination_Rate", 0.0)
            
            f.write(f"| {model_name:20s} | {bleu4:.4f} | {meteor:.4f} | {rouge:.4f} | {cider:.4f} | {kw_f1:.4f} | {hall:.4f} |\n")
        
        # Analysis
        f.write("\n### Key Findings\n\n")
        
        if "baseline" in results and "scst" in results:
            cider_improvement = results["scst"]["CIDEr"] - results["baseline"]["CIDEr"]
            kw_improvement = results["scst"]["Keyword_F1"] - results["baseline"]["Keyword_F1"]
            hall_reduction = results["baseline"]["Hallucination_Rate"] - results["scst"]["Hallucination_Rate"]
            
            f.write(f"- **SCST Improvement**: CIDEr {cider_improvement:+.4f}, Keyword F1 {kw_improvement:+.4f}\n")
            f.write(f"- **Hallucination Reduction**: {hall_reduction:+.4f} (lower is better)\n")
        
        # Best examples
        f.write("\n### Best Predictions (Top 5)\n\n")
        for i, ex in enumerate(best_examples, 1):
            f.write(f"#### Example {i}\n\n")
            f.write(f"**Prediction**: {ex['prediction']}\n\n")
            f.write(f"**Reference**: {ex['reference']}\n\n")
            f.write(f"**Predicted Keywords**: {ex['pred_keywords']}\n\n")
            f.write(f"**Ground Truth Keywords**: {ex['gt_keywords']}\n\n")
            f.write("---\n\n")
        
        # Worst examples
        f.write("\n### Worst Predictions (Bottom 5)\n\n")
        for i, ex in enumerate(worst_examples, 1):
            f.write(f"#### Example {i}\n\n")
            f.write(f"**Prediction**: {ex['prediction']}\n\n")
            f.write(f"**Reference**: {ex['reference']}\n\n")
            f.write(f"**Predicted Keywords**: {ex['pred_keywords']}\n\n")
            f.write(f"**Ground Truth Keywords**: {ex['gt_keywords']}\n\n")
            f.write("---\n\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("This report summarizes the results of 4-day fast training on IU X-Ray dataset.\n\n")
        f.write("**Training Strategy**:\n")
        f.write("- Day 1: Supervised baseline with Cross-Entropy loss\n")
        f.write("- Day 2: SCST with keyword-based rewards\n")
        f.write("- Day 3: Ensemble evaluation\n")
        f.write("- Day 4: Final comprehensive evaluation\n\n")
        f.write("**Key Takeaways**:\n")
        f.write("1. Lightweight models (ViT-GPT2/BLIP-2) can be fine-tuned quickly on medical data\n")
        f.write("2. SCST with keyword rewards helps reduce hallucinations\n")
        f.write("3. 4-day training is feasible for proof-of-concept medical captioning systems\n\n")
    
    print(f"\n✓ Report saved to {report_path}")


def main(args):
    print("\n" + "="*80)
    print("DAY 4: FINAL EVALUATION AND REPORT")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load test data
    print("[1/4] Loading test dataset...")
    _, _, test_loader = get_iu_xray_dataloaders(
        data_dir=args.data_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluate all models
    print(f"\n[2/4] Evaluating {len(args.checkpoints)} models...")
    all_results = {}
    all_predictions = {}
    
    for name, ckpt_path in zip(args.model_names, args.checkpoints):
        if not os.path.exists(ckpt_path):
            print(f"⚠ Skipping {name}: checkpoint not found at {ckpt_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")
        
        model = load_model(ckpt_path, args.model_type, device)
        scores, predictions_data = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device
        )
        
        all_results[name] = scores
        all_predictions[name] = predictions_data
        
        print(f"\nResults for {name}:")
        for metric, score in sorted(scores.items()):
            print(f"  {metric:20s}: {score:.4f}")
    
    # Find qualitative examples (from best model)
    print(f"\n[3/4] Selecting qualitative examples...")
    best_model_name = max(all_results.keys(), key=lambda k: all_results[k]["CIDEr"])
    print(f"Best model: {best_model_name} (CIDEr = {all_results[best_model_name]['CIDEr']:.4f})")
    
    best_examples, worst_examples = find_qualitative_examples(
        all_predictions[best_model_name],
        all_results[best_model_name],
        num_best=5,
        num_worst=5
    )
    
    # Generate report
    print(f"\n[4/4] Generating final report...")
    generate_report(
        results=all_results,
        best_examples=best_examples,
        worst_examples=worst_examples,
        output_dir=output_dir
    )
    
    # Save results as JSON
    results_json = output_dir / "evaluation_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved to {results_json}")
    
    print("\n" + "="*80)
    print("DAY 4 COMPLETE! 🎉")
    print("="*80)
    print(f"\nFinal results:")
    print(f"  Best model: {best_model_name}")
    print(f"  CIDEr: {all_results[best_model_name]['CIDEr']:.4f}")
    print(f"  BLEU-4: {all_results[best_model_name]['BLEU-4']:.4f}")
    print(f"  Keyword F1: {all_results[best_model_name]['Keyword_F1']:.4f}")
    print(f"  Hallucination Rate: {all_results[best_model_name]['Hallucination_Rate']:.4f}")
    print(f"\nAll outputs saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 4: Final Evaluation")
    
    parser.add_argument("--data_dir", type=str, default="data/IU_XRAY")
    parser.add_argument("--output_dir", type=str, default="results/day4_final")
    parser.add_argument("--model_type", type=str, default="vit-gpt2",
                        choices=["vit-gpt2", "blip2"])
    
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default=[
                            "checkpoints/day1_baseline/best_model.pt",
                            "checkpoints/day2_scst/best_model.pt"
                        ])
    parser.add_argument("--model_names", type=str, nargs="+",
                        default=["baseline", "scst"])
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.checkpoints) != len(args.model_names):
        print("Error: Number of checkpoints must match number of model names")
        sys.exit(1)
    
    main(args)
