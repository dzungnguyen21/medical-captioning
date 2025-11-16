# Fast Medical Image Captioning 🚀
## 4-Day Training Pipeline for Kaggle T4 x2 GPUs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This is a **fast, practical approach** to medical image captioning optimized for **resource-constrained environments** (Kaggle, Colab, etc.). 

Unlike the full research implementation, this focuses on:
- ✅ **4-day training timeline**
- ✅ **2x T4 GPU (16GB each)**
- ✅ **Pre-trained models** (BLIP-2, ViT-GPT2)
- ✅ **LoRA fine-tuning** (parameter-efficient)
- ✅ **Keyword-based rewards** (fast alternative to RadGraph)
- ✅ **Hallucination reduction** via SCST

---

## 📊 Key Differences from Full Implementation

| Aspect | Full Implementation | Fast Implementation |
|--------|---------------------|---------------------|
| **Training Time** | 2-3 weeks | **4 days** |
| **GPU Requirement** | Multiple A100s | **2x T4** |
| **Dataset** | MS-COCO + MIMIC-CXR | **IU X-Ray only** |
| **Architecture** | Custom Faster R-CNN + Transformer | **Pre-trained BLIP-2/ViT-GPT2** |
| **Region Encoding** | Bottom-up attention | **Vision Transformer** |
| **Hallucination Detection** | RadGraph + CheXbert | **Keyword matching** |
| **Training Strategy** | Full fine-tuning | **LoRA + 8-bit** |
| **Expected CIDEr** | 0.8-1.2 | **0.5-0.6** |

---

## 🗂️ Project Structure

```
medical_img_captioning_train/
├── Fast_Models/                    # Lightweight model wrappers
│   ├── blip2_wrapper.py           # BLIP-2 with LoRA/8-bit (recommended)
│   └── vit_gpt2_wrapper.py        # ViT-GPT2 (faster alternative)
│
├── Fast_Data/                      # Optimized data loaders
│   └── iu_xray_loader.py          # IU X-Ray dataset
│
├── Fast_Rewards/                   # Reward functions for RL
│   └── keyword_reward.py          # CIDEr + Keyword F1 - Hallucination
│
├── Fast_Training/                  # Training utilities
│   └── trainer.py                 # CE + SCST trainer with fp16
│
├── 4Day_Scripts/                   # Day-by-day training scripts
│   ├── day1_baseline.py           # Day 1: Cross-Entropy baseline
│   ├── day2_scst.py               # Day 2: SCST with keyword rewards
│   ├── day3_ensemble.py           # Day 3: Ensemble evaluation
│   └── day4_evaluate.py           # Day 4: Final comprehensive evaluation
│
├── KAGGLE_QUICKSTART.md           # Detailed 4-day guide
├── requirements_fast.txt           # Dependencies
└── README_FAST.md                 # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-repo/medical-captioning.git
cd medical-captioning

# Install dependencies
pip install -r requirements_fast.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### 2. Data Preparation

**Download IU X-Ray dataset**:

```bash
# Create data directory
mkdir -p data/IU_XRAY

# Download from OpenI (https://openi.nlm.nih.gov/)
# Or use Kaggle dataset if available

# Expected structure:
# data/IU_XRAY/
#   ├── images/          # ~7,470 PNG images
#   └── indiana_reports.csv
```

### 3. Day 1: Baseline Training (4-6 hours)

```bash
# Train with ViT-GPT2 (recommended for speed)
python 4Day_Scripts/day1_baseline.py \
    --data_dir data/IU_XRAY \
    --model_type vit-gpt2 \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    --use_lora \
    --use_fp16 \
    --use_multi_gpu

# Or with BLIP-2 (better quality, slower)
python 4Day_Scripts/day1_baseline.py \
    --model_type blip2 \
    --batch_size 8 \
    --use_8bit
```

**Expected output**:
- Checkpoint: `checkpoints/day1_baseline/best_model.pt`
- CIDEr: ~0.30-0.50
- BLEU-4: ~0.15-0.20

### 4. Day 2: SCST Training (6-8 hours)

```bash
python 4Day_Scripts/day2_scst.py \
    --data_dir data/IU_XRAY \
    --model_type vit-gpt2 \
    --pretrained_checkpoint checkpoints/day1_baseline/best_model.pt \
    --epochs 5 \
    --batch_size 8 \
    --lr 1e-6 \
    --cider_weight 1.0 \
    --keyword_weight 0.5 \
    --hallucination_penalty 0.3
```

**Expected improvement**:
- CIDEr: +0.05-0.15
- Keyword F1: +5-10%
- Hallucination: -10-20%

### 5. Day 3: Ensemble (2-4 hours)

```bash
python 4Day_Scripts/day3_ensemble.py \
    --checkpoints \
        checkpoints/day1_baseline/best_model.pt \
        checkpoints/day2_scst/best_model.pt \
    --model_names baseline scst
```

### 6. Day 4: Final Evaluation (2-4 hours)

```bash
python 4Day_Scripts/day4_evaluate.py \
    --checkpoints \
        checkpoints/day1_baseline/best_model.pt \
        checkpoints/day2_scst/best_model.pt \
    --model_names baseline scst \
    --output_dir results/final
```

**Outputs**:
- `results/final/FINAL_REPORT.md`
- `results/final/evaluation_results.json`
- Best/worst examples for qualitative analysis

---

## 📈 Expected Results

### Quantitative Metrics

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr | Keyword F1 | Hallucination |
|-------|--------|--------|---------|-------|------------|---------------|
| **Baseline (CE)** | 0.18 | 0.24 | 0.42 | 0.45 | 0.62 | 0.28 |
| **SCST (RL)** | 0.20 | 0.26 | 0.44 | 0.52 | 0.68 | 0.21 |
| **Ensemble** | 0.21 | 0.27 | 0.45 | 0.54 | 0.70 | 0.19 |

### Sample Outputs

**Input**: Chest X-ray image

**Baseline**: "The lungs are clear. Heart size is normal."

**SCST**: "No acute cardiopulmonary abnormality. Lungs are clear bilaterally."

**Ground Truth**: "Clear lungs. Normal heart size. No pleural effusion."

---

## 💡 Key Innovations

### 1. Keyword-Based Reward Function

Instead of expensive RadGraph, we use **fast keyword matching**:

```python
Reward = α × CIDEr + β × Keyword_F1 - γ × Hallucination_Penalty

where:
- CIDEr: Language quality (from pycocoevalcap)
- Keyword_F1: F1 between predicted and GT medical keywords
- Hallucination_Penalty: % of keywords not in ground truth
```

**Medical keywords** (~200 terms):
- Anatomy: lung, heart, mediastinum, diaphragm, etc.
- Pathology: pneumonia, effusion, consolidation, etc.
- Descriptors: bilateral, mild, severe, etc.

### 2. LoRA Fine-Tuning

Only trains **0.5-2% of parameters**:
- Baseline: 110M parameters → 1.5M trainable (LoRA)
- BLIP-2: 2.7B parameters → 5M trainable (LoRA)

**Benefits**:
- 10x faster training
- 5x less memory
- No catastrophic forgetting

### 3. Mixed Precision (FP16)

- 2x faster training
- 2x less VRAM usage
- Minimal accuracy loss

---

## 🛠️ Advanced Usage

### Custom Medical Keywords

```python
from Fast_Rewards.keyword_reward import KeywordRewardFunction

custom_keywords = {
    "covid", "coronavirus", "ground glass", "consolidation"
}

reward_fn = KeywordRewardFunction(
    custom_keywords=custom_keywords,
    cider_weight=1.0,
    keyword_weight=0.7,  # Increase weight for domain-specific terms
    hallucination_penalty=0.5
)
```

### Gradient Accumulation (for larger effective batch size)

```bash
python 4Day_Scripts/day1_baseline.py \
    --batch_size 4 \
    --gradient_accumulation_steps 4  # Effective batch size = 16
```

### Early Stopping for SCST

```python
# In day2_scst.py, modify training loop:
if scst_stats['avg_reward'] < -1.0:
    print("RL diverging! Stopping early.")
    break
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Reduce batch size
--batch_size 4

# Use gradient checkpointing (in trainer.py)
# Enable in model wrapper

# Or reduce sequence length
--max_length 64
```

### Issue: RL Rewards Dropping

**Solution**:
```bash
# Lower learning rate
--lr 5e-7

# Reduce number of epochs
--epochs 3

# Or stop training and use Day 1 checkpoint
```

### Issue: Training Too Slow

**Solution**:
```bash
# Use ViT-GPT2 instead of BLIP-2
--model_type vit-gpt2

# Reduce evaluation frequency
--eval_every 5

# Enable fp16 (if not already)
--use_fp16
```

---

## 📚 References

1. **BLIP-2**: Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training" (2023)
2. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation" (2021)
3. **SCST**: Rennie et al. "Self-Critical Sequence Training" (2017)
4. **IU X-Ray**: Demner-Fushman et al. "Preparing a collection of radiology examinations" (2016)

---

## 🤝 Contributing

Improvements welcome! Especially:
- Better keyword lists for different medical domains
- Alternative reward functions
- More efficient training strategies
- Additional datasets (ChestX-ray14, etc.)

---

## 📝 License

MIT License - see `LICENSE` file

---

## 🙏 Acknowledgments

- Hugging Face for Transformers & PEFT
- Microsoft for COCO evaluation tools
- OpenI for IU X-Ray dataset
- Kaggle for GPU resources

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Check `KAGGLE_QUICKSTART.md` for detailed instructions
- Review `TROUBLESHOOTING.md` for common problems

---

**Happy training! 🎉**

Remember: This is a **fast, practical implementation**. For state-of-the-art results, see the full research implementation.
