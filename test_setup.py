"""
Test script to verify all modules are working correctly
"""

import sys
import os

print("=" * 80)
print("TESTING IMAGE CAPTIONING SYSTEM")
print("=" * 80)

# Test 1: Import core libraries
print("\n[1/7] Testing core library imports...")
try:
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Import NLP libraries
print("\n[2/7] Testing NLP library imports...")
try:
    import transformers
    import nltk
    from pycocoevalcap.cider.cider import Cider
    print(f"✓ Transformers version: {transformers.__version__}")
    print("✓ NLTK available")
    print("✓ COCO eval tools available")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Import custom modules
print("\n[3/7] Testing custom module imports...")
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from Shared_Modules.region_encoder import RegionFeatureExtractor
    from Shared_Modules.transformer_decoder import RegionAwareTransformerDecoder
    from Shared_Modules.trainer import CaptioningModel
    from Shared_Modules.hallucination_detector import HallucinationDetector
    from Shared_Modules.reward_functions import GeneralReward
    from Shared_Modules.metrics import CaptionMetrics
    
    print("✓ Region encoder module")
    print("✓ Transformer decoder module")
    print("✓ Training module")
    print("✓ Hallucination detector module")
    print("✓ Reward functions module")
    print("✓ Metrics module")
except Exception as e:
    print(f"✗ Error importing custom modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create dummy vocabulary
print("\n[4/7] Creating dummy vocabulary...")
try:
    vocab = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3,
        'a': 4,
        'dog': 5,
        'cat': 6,
        'is': 7,
        'running': 8,
        'sitting': 9,
        'on': 10,
        'the': 11,
        'grass': 12
    }
    print(f"✓ Vocabulary created with {len(vocab)} tokens")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 5: Initialize models
print("\n[5/7] Initializing models...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create region encoder
    region_encoder = RegionFeatureExtractor(
        pretrained=False,  # Don't download weights for testing
        num_regions=10,
        feature_dim=256,  # Smaller for testing
        device=device
    ).to(device)
    print("✓ Region encoder initialized")
    
    # Create decoder
    decoder = RegionAwareTransformerDecoder(
        vocab_size=len(vocab),
        d_model=128,  # Smaller for testing
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_len=20,
        dropout=0.1,
        pad_token_id=vocab['<PAD>']
    ).to(device)
    print("✓ Transformer decoder initialized")
    
    # Create full model
    model = CaptioningModel(
        region_encoder=region_encoder,
        decoder=decoder,
        region_feature_dim=256,
        decoder_dim=128
    ).to(device)
    print("✓ Full captioning model initialized")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"✗ Error initializing models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test forward pass
print("\n[6/7] Testing forward pass...")
try:
    batch_size = 2
    
    # Create dummy input
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_captions = torch.randint(0, len(vocab), (batch_size, 15)).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits, attention, encoder_output = model(dummy_images, dummy_captions)
    
    print(f"✓ Forward pass successful")
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Expected shape: ({batch_size}, 15, {len(vocab)})")
    
    assert logits.shape == (batch_size, 15, len(vocab)), "Output shape mismatch"
    
except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test generation
print("\n[7/7] Testing caption generation...")
try:
    # Generate captions
    with torch.no_grad():
        generated_ids, attention, encoder_output = model.generate(
            dummy_images,
            start_token_id=vocab['<START>'],
            end_token_id=vocab['<END>'],
            max_len=10,
            beam_size=1  # Greedy
        )
    
    print(f"✓ Generation successful")
    print(f"✓ Generated shape: {generated_ids.shape}")
    
    # Decode one caption
    idx_to_word = {v: k for k, v in vocab.items()}
    sample_caption = []
    for token_id in generated_ids[0].cpu().numpy():
        if token_id == vocab['<END>']:
            break
        if token_id in idx_to_word:
            sample_caption.append(idx_to_word[token_id])
    
    print(f"✓ Sample generated caption: {' '.join(sample_caption)}")
    
except Exception as e:
    print(f"✗ Error in generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour environment is ready for image captioning training!")
print("\nNext steps:")
print("1. Download MS-COCO or MIMIC-CXR dataset")
print("2. Run training script: python General_Domain/train_general.py")
print("3. Evaluate results: python General_Domain/evaluate_general.py")
print("\nFor more information, see README.md and QUICKSTART.md")
print("=" * 80)
