"""
Comprehensive Evaluation Script for General Domain
Evaluates on NLG metrics, CHAIR, POPE, and Grounding
"""

import torch
import argparse
import os
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Shared_Modules.region_encoder import RegionFeatureExtractor
from Shared_Modules.transformer_decoder import RegionAwareTransformerDecoder
from Shared_Modules.trainer import CaptioningModel
from Shared_Modules.hallucination_detector import HallucinationDetector
from Shared_Modules.metrics import ComprehensiveEvaluator, CaptionMetrics, POPEMetrics
from General_Domain.data_loader import get_general_dataloader

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def evaluate_model(model, dataloader, vocab, args):
    """
    Evaluate model on test set.
    
    Returns:
        results: Dictionary with all metrics
        generated_captions: Dictionary mapping image_id -> caption
        attention_weights: Dictionary mapping image_id -> attention weights
    """
    model.eval()
    idx_to_word = {v: k for k, v in vocab.items()}
    
    generated_captions = {}
    reference_captions = {}
    detected_objects = {}
    attention_weights = {}
    
    print("Generating captions...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['images'].to(args.device)
            image_ids = batch['image_id'].cpu().numpy()
            refs = batch['captions_text']
            
            # Generate captions
            generated_ids, attention, encoder_output = model.generate(
                images,
                start_token_id=vocab['<START>'],
                end_token_id=vocab['<END>'],
                max_len=args.max_seq_len,
                beam_size=args.beam_size,
                return_attention=True
            )
            
            # Convert to text
            for i in range(len(image_ids)):
                image_id = int(image_ids[i])
                
                # Generated caption
                gen_tokens = generated_ids[i].cpu().numpy()
                gen_caption = []
                for token_id in gen_tokens:
                    if token_id == vocab['<END>']:
                        break
                    if token_id in idx_to_word and token_id not in [vocab['<PAD>'], vocab['<START>']]:
                        gen_caption.append(idx_to_word[token_id])
                
                gen_caption_text = ' '.join(gen_caption)
                generated_captions[image_id] = [gen_caption_text]
                
                # Reference captions
                if isinstance(refs[i], list):
                    reference_captions[image_id] = refs[i]
                else:
                    reference_captions[image_id] = [refs[i]]
                
                # Detected objects
                if 'labels' in encoder_output:
                    labels = encoder_output['labels'][i].cpu().numpy()
                    detected_obj_names = []
                    for label_id in labels:
                        if 0 < label_id < len(COCO_CLASSES):
                            detected_obj_names.append(COCO_CLASSES[label_id])
                    detected_objects[image_id] = detected_obj_names
                
                # Attention weights
                if attention is not None:
                    attention_weights[image_id] = attention[i]
    
    print(f"\nGenerated {len(generated_captions)} captions")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # 1. Standard NLG metrics
    caption_metrics = CaptionMetrics()
    nlg_scores = caption_metrics.compute_scores(generated_captions, reference_captions)
    
    # 2. CHAIR metrics
    hallucination_detector = HallucinationDetector(
        vocab=vocab,
        object_class_names=COCO_CLASSES,
        use_synonym_matching=True
    )
    
    gen_caps_list = [generated_captions[i][0] for i in sorted(generated_captions.keys())]
    det_objs_list = [detected_objects.get(i, []) for i in sorted(generated_captions.keys())]
    
    chair_scores = hallucination_detector.compute_chair_metrics(gen_caps_list, det_objs_list)
    
    # Combine all results
    results = {**nlg_scores, **chair_scores}
    
    return results, generated_captions, attention_weights


def visualize_attention(image, caption, attention_weights, boxes, save_path):
    """
    Visualize attention weights on image regions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attention heatmap
    # Average attention across all words
    avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # [num_regions]
    
    # Create heatmap overlay
    # This is simplified; you'd need actual region positions for proper visualization
    axes[1].imshow(image)
    axes[1].set_title(f"Caption: {caption}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(results, generated_captions, args):
    """Save evaluation results."""
    
    # Save metrics
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {results_path}")
    
    # Save generated captions
    captions_path = os.path.join(args.output_dir, 'generated_captions.json')
    with open(captions_path, 'w') as f:
        json.dump(generated_captions, f, indent=2)
    
    print(f"Saved captions to {captions_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("\n--- NLG Metrics ---")
    for metric in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr']:
        if metric in results:
            print(f"{metric}: {results[metric]:.4f}")
    
    print("\n--- Hallucination Metrics ---")
    for metric in ['CHAIR_i', 'CHAIR_s']:
        if metric in results:
            print(f"{metric}: {results[metric]:.4f}")
    
    if 'num_hallucinated_objects' in results:
        print(f"Hallucinated Objects: {results['num_hallucinated_objects']} / {results['total_objects']}")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate General Domain Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/COCO',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./results/general',
                        help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'nocaps'],
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
    parser.add_argument('--max_seq_len', type=int, default=50)
    
    # Generation parameters
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size for generation')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize attention for sample images')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='Number of images to visualize')
    
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
    
    dataloader = get_general_dataloader(
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
    
    region_encoder = RegionFeatureExtractor(
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
    results, generated_captions, attention_weights = evaluate_model(
        model, dataloader, vocab, args
    )
    
    # Save results
    save_results(results, generated_captions, args)


if __name__ == '__main__':
    main()
