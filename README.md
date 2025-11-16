# Region-Aware Image Captioning with Hallucination Mitigation

Hệ thống Image Captioning hai luồng (General & Medical) với cơ chế chống Hallucination sử dụng Reinforcement Learning và Region-based Grounding.

## 📋 Tổng Quan

Dự án này thực hiện chiến lược huấn luyện hai giai đoạn:
1. **Phase 1: Supervised Learning** - Cross-Entropy Loss với Teacher Forcing
2. **Phase 2: Reinforcement Learning** - Self-Critical Sequence Training (SCST) với custom reward functions

### Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Region Encoder (Faster R-CNN / Medical-specific Detector)  │
│  - Extracts region features (bottom-up attention)           │
│  - Outputs: [num_regions, feature_dim]                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Transformer Decoder with Cross-Attention                   │
│  - Attends to region features                               │
│  - Generates captions autoregressively                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Hallucination Detection Module                             │
│  - Compares generated words with detected objects           │
│  - Penalizes hallucinated entities                          │
└─────────────────────────────────────────────────────────────┘
```

## 🗂️ Cấu Trúc Dự Án

```
medical_img_captioning_train/
│
├── Shared_Modules/              # Modules dùng chung
│   ├── region_encoder.py        # Faster R-CNN region extractor
│   ├── transformer_decoder.py   # Transformer decoder với attention
│   ├── hallucination_detector.py # CHAIR & RadGraph
│   ├── trainer.py               # Supervised & RL trainers
│   ├── reward_functions.py      # CIDEr, CHAIR penalty, RadGraph F1
│   └── metrics.py               # Comprehensive evaluation metrics
│
├── General_Domain/              # Luồng Ảnh Đa dụng
│   ├── data_loader.py           # MS-COCO, Visual Genome, NoCaps
│   ├── train_general.py         # Training script
│   └── evaluate_general.py      # Evaluation script
│
├── Medical_Domain/              # Luồng Ảnh Y tế
│   ├── data_loader.py           # MIMIC-CXR, VinDr-CXR, IU X-Ray
│   ├── train_medical.py         # Training script
│   └── evaluate_medical.py      # Evaluation script
│
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Cài Đặt

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cài đặt Detectron2 (cho Faster R-CNN)

```bash
# CUDA 11.7
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html
```

### 3. Cài đặt COCO Evaluation API

```bash
pip install pycocoevalcap
```

### 4. (Optional) Medical-specific tools

```bash
# RadGraph
pip install radgraph

# CheXbert
pip install chexbert
```

## 📊 Dữ Liệu

### Luồng General

| Dataset | Mục đích | Link |
|---------|----------|------|
| **MS-COCO** (Karpathy Split) | Train/Val | [COCO](https://cocodataset.org/) |
| **Visual Genome** | Region pre-training | [Visual Genome](https://visualgenome.org/) |
| **NoCaps** | Test (Hallucination) | [NoCaps](https://nocaps.org/) |

### Luồng Medical

| Dataset | Mục đích | Link |
|---------|----------|------|
| **MIMIC-CXR** | Train/Val | [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) |
| **VinDr-CXR** | Region annotations | [VinDr-CXR](https://vindr.ai/datasets/cxr) |
| **IU X-Ray** | Test (Cross-dataset) | [IU X-Ray](https://openi.nlm.nih.gov/) |

## 🎯 Training

### General Domain

```bash
# Phase 1: Supervised Training
python General_Domain/train_general.py \
    --data_dir ./data/COCO \
    --checkpoint_dir ./checkpoints/general \
    --epochs_xe 30 \
    --lr_xe 5e-4 \
    --batch_size 32 \
    --skip_rl

# Phase 2: RL Training
python General_Domain/train_general.py \
    --data_dir ./data/COCO \
    --checkpoint_dir ./checkpoints/general \
    --epochs_rl 20 \
    --lr_rl 1e-5 \
    --reward_cider_weight 1.0 \
    --reward_chair_weight 1.0 \
    --skip_xe
```

### Medical Domain

```bash
# Phase 1: Supervised Training
python Medical_Domain/train_medical.py \
    --data_dir ./data/MIMIC-CXR \
    --checkpoint_dir ./checkpoints/medical \
    --epochs_xe 50 \
    --lr_xe 5e-4 \
    --batch_size 16 \
    --max_seq_len 200 \
    --skip_rl

# Phase 2: RL Training (với RadGraph)
python Medical_Domain/train_medical.py \
    --data_dir ./data/MIMIC-CXR \
    --checkpoint_dir ./checkpoints/medical \
    --epochs_rl 30 \
    --lr_rl 5e-6 \
    --reward_cider_weight 1.0 \
    --reward_radgraph_weight 2.0 \
    --use_radgraph \
    --skip_xe
```

## 📈 Evaluation

### General Domain

```bash
python General_Domain/evaluate_general.py \
    --checkpoint ./checkpoints/general/best_rl_model.pth \
    --data_dir ./data/COCO \
    --output_dir ./results/general \
    --dataset coco \
    --split test \
    --beam_size 3
```

**Metrics được báo cáo:**
- **NLG Standard:** BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
- **Hallucination:** CHAIR_i, CHAIR_s
- **Grounding:** Pointing Game Accuracy (if attention visualization enabled)

### Medical Domain

```bash
python Medical_Domain/evaluate_medical.py \
    --checkpoint ./checkpoints/medical/best_rl_model.pth \
    --data_dir ./data/MIMIC-CXR \
    --output_dir ./results/medical \
    --dataset mimic_cxr \
    --split test \
    --beam_size 3 \
    --use_radgraph \
    --use_chexbert
```

**Metrics được báo cáo:**
- **NLG Standard:** BLEU, METEOR, ROUGE-L, CIDEr
- **Medical:** RadGraph F1, CheXbert F1 (14 pathologies)

## 🎨 Reward Functions

### General Domain

$$
R_{\text{total}} = \alpha \cdot \text{CIDEr} + \beta \cdot (1 - \text{CHAIR}_i)
$$

- **CIDEr**: Đo độ trôi chảy và tương đồng với human captions
- **CHAIR_i**: Penalize hallucinated objects
- Mặc định: α = 1.0, β = 1.0

### Medical Domain

$$
R_{\text{total}} = \alpha \cdot \text{CIDEr} + \beta \cdot \text{RadGraph F1}
$$

- **CIDEr**: Đo độ trôi chảy
- **RadGraph F1**: Đo độ chính xác về clinical entities và relations
- Mặc định: α = 1.0, β = 2.0 (ưu tiên clinical accuracy)

## 📊 Kết Quả Mong Đợi

### General Domain (MS-COCO)

| Model | BLEU-4 | CIDEr | CHAIR_i ↓ | CHAIR_s ↓ |
|-------|--------|-------|-----------|-----------|
| Baseline (Up-Down) | 36.2 | 120.1 | 8.3 | 18.2 |
| **Ours (XE)** | 36.5 | 121.3 | 7.8 | 17.5 |
| **Ours (RL)** | **37.8** | **126.7** | **5.2** | **12.1** |

### Medical Domain (MIMIC-CXR)

| Model | BLEU-4 | CIDEr | RadGraph F1 |
|-------|--------|-------|-------------|
| Baseline | 14.2 | 35.6 | 0.312 |
| **Ours (XE)** | 14.8 | 37.1 | 0.325 |
| **Ours (RL)** | **15.3** | **39.8** | **0.361** |

## 🔬 Ablation Studies

Để chạy ablation studies, điều chỉnh reward weights:

```bash
# Chỉ dùng CIDEr (baseline SCST)
python General_Domain/train_general.py \
    --reward_cider_weight 1.0 \
    --reward_chair_weight 0.0

# Chỉ dùng CHAIR penalty
python General_Domain/train_general.py \
    --reward_cider_weight 0.0 \
    --reward_chair_weight 1.0

# Cân bằng
python General_Domain/train_general.py \
    --reward_cider_weight 1.0 \
    --reward_chair_weight 1.0
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Giảm batch size
--batch_size 8

# Giảm số regions
--num_regions 20

# Giảm model size
--d_model 256 --num_layers 4
```

### Detectron2 Installation Issues
```bash
# Build from source
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

### RadGraph Not Available
```bash
# RadGraph requires specific dependencies
pip install radgraph --no-deps
pip install spacy scispacy
python -m spacy download en_core_sci_sm
```

## 📝 Citation

Nếu sử dụng code này trong nghiên cứu, vui lòng cite:

```bibtex
@article{your_paper,
  title={Region-Aware Image Captioning with Hallucination Mitigation via Reinforcement Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 📚 References

1. **SCST**: Rennie et al. "Self-critical Sequence Training for Image Captioning" (CVPR 2017)
2. **Bottom-Up Attention**: Anderson et al. "Bottom-Up and Top-Down Attention for Image Captioning" (CVPR 2018)
3. **CHAIR**: Rohrbach et al. "Object Hallucination in Image Captioning" (EMNLP 2018)
4. **RadGraph**: Jain et al. "RadGraph: Extracting Clinical Entities and Relations from Radiology Reports" (NeurIPS 2021)
5. **MIMIC-CXR**: Johnson et al. "MIMIC-CXR: A large publicly available database of labeled chest radiographs" (2019)

## 🙏 Acknowledgments

- MS-COCO dataset team
- MIMIC-CXR dataset creators
- PyTorch và Hugging Face communities
- Detectron2 team

## 📧 Contact

Nếu có câu hỏi, vui lòng tạo issue hoặc liên hệ qua email.

---

**License**: MIT

**Last Updated**: November 2025
