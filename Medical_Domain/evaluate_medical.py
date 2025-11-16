"""
Comprehensive Evaluation Script for Medical Domain
Evaluates on NLG metrics, RadGraph F1, CheXbert F1
"""

import torch
import argparse
import os
import sys
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Shared_Modules.region_encoder import MedicalRegionExtractor
from Shared_Modules.transformer_decoder import RegionAwareTransformerDecoder
from Shared_Modules.trainer import CaptioningModel
from Shared_Modules.hallucination_detector import MedicalHallucinationDetector
from Shared_Modules.metrics import CaptionMetrics, MedicalMetrics
from Medical_Domain.data_loader import get_medical_dataloader


def evaluate_model(model, dataloader, vocab, args):
    """
    Evaluate model on test set.
    
    Returns:
        results: Dictionary with all metrics
        generated_reports: Dictionary mapping image_id -> report
    """
    model.eval()
    idx_to_word = {v: k for k, v in vocab.items()}
    
    generated_reports = {}
    reference_reports = {}
    
    print("Generating reports...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['images'].to(args.device)
            image_ids = batch['image_id'].cpu().numpy()
            refs = batch['captions_text']
            
            # Generate reports
            generated_ids, _, _ = model.generate(
                images,
                start_token_id=vocab['<START>'],
                end_token_id=vocab['<END>'],
                max_len=args.max_seq_len,
                beam_size=args.beam_size,
                return_attention=False
            )
            
            # Convert to text
            for i in range(len(image_ids)):
                image_id = int(image_ids[i])
                
                # Generated report
                gen_tokens = generated_ids[i].cpu().numpy()
                gen_report = []
                for token_id in gen_tokens:
                    if token_id == vocab['<END>']:
                        break
                    if token_id in idx_to_word and token_id not in [vocab['<PAD>'], vocab['<START>']]:
                        gen_report.append(idx_to_word[token_id])
                
                gen_report_text = ' '.join(gen_report)
                generated_reports[image_id] = [gen_report_text]
                
                # Reference report
                reference_reports[image_id] = [refs[i]]
    
    print(f"\nGenerated {len(generated_reports)} reports")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # 1. Standard NLG metrics
    caption_metrics = CaptionMetrics()
    nlg_scores = caption_metrics.compute_scores(generated_reports, reference_reports)
    
    # 2. Medical-specific metrics
    medical_metrics = MedicalMetrics(
        use_radgraph=args.use_radgraph,
        use_chexbert=args.use_chexbert
    )
    
    # RadGraph F1
    radgraph_scores = {}
    if args.use_radgraph:
        gen_reports_list = [generated_reports[i][0] for i in sorted(generated_reports.keys())]
        ref_reports_list = [reference_reports[i][0] for i in sorted(reference_reports.keys())]
        
        radgraph_scores = medical_metrics.compute_radgraph_f1(
            gen_reports_list, ref_reports_list
        )
    
    # Combine all results
    results = {**nlg_scores, **radgraph_scores}
    
    return results, generated_reports


def analyze_clinical_entities(generated_reports, reference_reports):
    """
    Analyze clinical entity distribution in generated vs reference reports.
    """
    anatomical_entities = {
        'lung', 'lungs', 'heart', 'cardiac', 'chest', 'thorax',
        'mediastinum', 'diaphragm', 'pleura', 'pleural'
    }
    
    pathology_entities = {
        'pneumonia', 'effusion', 'edema', 'atelectasis', 'consolidation',
        'pneumothorax', 'cardiomegaly', 'nodule', 'mass', 'opacity'
    }
    
    gen_anatomical = []
    gen_pathology = []
    ref_anatomical = []
    ref_pathology = []
    
    for image_id in generated_reports:
        gen_text = generated_reports[image_id][0].lower()
        ref_text = reference_reports[image_id][0].lower()
        
        # Count entities
        gen_anat = sum(1 for e in anatomical_entities if e in gen_text)
        gen_path = sum(1 for e in pathology_entities if e in gen_text)
        ref_anat = sum(1 for e in anatomical_entities if e in ref_text)
        ref_path = sum(1 for e in pathology_entities if e in ref_text)
        
        gen_anatomical.append(gen_anat)
        gen_pathology.append(gen_path)
        ref_anatomical.append(ref_anat)
        ref_pathology.append(ref_path)
    
    analysis = {
        'avg_gen_anatomical': np.mean(gen_anatomical),
        'avg_gen_pathology': np.mean(gen_pathology),
        'avg_ref_anatomical': np.mean(ref_anatomical),
        'avg_ref_pathology': np.mean(ref_pathology)
    }
    
    return analysis


def save_results(results, generated_reports, args):
    """Save evaluation results."""
    
    # Save metrics
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {results_path}")
    
    # Save generated reports
    reports_path = os.path.join(args.output_dir, 'generated_reports.json')
    with open(reports_path, 'w') as f:
        json.dump(generated_reports, f, indent=2)
    
    print(f"Saved reports to {reports_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - MEDICAL DOMAIN")
    print("=" * 80)
    
    print("\n--- NLG Metrics ---")
    for metric in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr']:
        if metric in results:
            print(f"{metric}: {results[metric]:.4f}")
    
    print("\n--- Medical Metrics ---")
    for metric in ['RadGraph_F1', 'RadGraph_Precision', 'RadGraph_Recall']:
        if metric in results:
            print(f"{metric}: {results[metric]:.4f}")
    
    for metric in results:
        if metric.startswith('CheXbert'):
            print(f"{metric}: {results[metric]:.4f}")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Medical Domain Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/MIMIC-CXR',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./results/medical',
                        help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='mimic_cxr',
                        choices=['mimic_cxr', 'iu_xray'],
                        help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate')
    
    # Model parameters
    parser.add_argument('--region_feature_dim', type=int, default=2048)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--num_regions', type=int, default=36)
    parser.add_argument('--max_seq_len', type=int, default=200)
    
    # Generation parameters
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size for generation')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Medical-specific
    parser.add_argument('--use_radgraph', action='store_true',
                        help='Compute RadGraph F1')
    parser.add_argument('--use_chexbert', action='store_true',
                        help='Compute CheXbert F1')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    vocab = checkpoint['vocab']
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    
    dataloader = get_medical_dataloader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        vocab=vocab
    )
    
    print(f"Test samples: {len(dataloader.dataset)}")
    
    # Build model
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # Evaluate
    results, generated_reports = evaluate_model(
        model, dataloader, vocab, args
    )
    
    # Save results
    save_results(results, generated_reports, args)


if __name__ == '__main__':
    main()
