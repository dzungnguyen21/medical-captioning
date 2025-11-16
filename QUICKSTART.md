# Quick Start Guide

Hướng dẫn nhanh để bắt đầu với hệ thống Image Captioning.

## 📦 Bước 1: Cài đặt môi trường

```bash
# Clone repository
cd d:\AI\medical_img_captioning_train

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html

# Cài đặt COCO evaluation tools
pip install pycocoevalcap
```

## 📊 Bước 2: Chuẩn bị dữ liệu

### General Domain (MS-COCO)

```bash
# Tải MS-COCO dataset
mkdir -p data/COCO
cd data/COCO

# Download images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

# Unzip
unzip train2014.zip
unzip val2014.zip

# Download Karpathy split
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
```

### Medical Domain (MIMIC-CXR)

```bash
# MIMIC-CXR requires PhysioNet credentialing
# Visit: https://physionet.org/content/mimic-cxr/2.0.0/

# After obtaining access:
mkdir -p data/MIMIC-CXR
cd data/MIMIC-CXR

# Download using wget with credentials
wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
```

## 🚀 Bước 3: Training

### Option A: Training từ đầu (General Domain)

```bash
# Phase 1: Supervised Training (30 epochs, ~12 hours on single GPU)
python General_Domain/train_general.py \
    --data_dir ./data/COCO \
    --checkpoint_dir ./checkpoints/general \
    --epochs_xe 30 \
    --batch_size 32 \
    --device cuda \
    --skip_rl

# Phase 2: RL Training (20 epochs, ~8 hours)
python General_Domain/train_general.py \
    --data_dir ./data/COCO \
    --checkpoint_dir ./checkpoints/general \
    --epochs_rl 20 \
    --batch_size 32 \
    --device cuda \
    --skip_xe
```

### Option B: Training từ đầu (Medical Domain)

```bash
# Phase 1: Supervised Training (50 epochs, ~24 hours)
python Medical_Domain/train_medical.py \
    --data_dir ./data/MIMIC-CXR \
    --checkpoint_dir ./checkpoints/medical \
    --epochs_xe 50 \
    --batch_size 16 \
    --device cuda \
    --skip_rl

# Phase 2: RL Training with RadGraph (30 epochs, ~15 hours)
python Medical_Domain/train_medical.py \
    --data_dir ./data/MIMIC-CXR \
    --checkpoint_dir ./checkpoints/medical \
    --epochs_rl 30 \
    --batch_size 16 \
    --use_radgraph \
    --device cuda \
    --skip_xe
```

### Option C: Training với config files

```bash
# Sử dụng YAML configs
python General_Domain/train_general.py --config configs/general_config.yaml
python Medical_Domain/train_medical.py --config configs/medical_config.yaml
```

## 📈 Bước 4: Evaluation

### Evaluate General Domain

```bash
python General_Domain/evaluate_general.py \
    --checkpoint ./checkpoints/general/best_rl_model.pth \
    --data_dir ./data/COCO \
    --output_dir ./results/general \
    --dataset coco \
    --split test \
    --beam_size 3
```

**Kết quả mong đợi:**
```
BLEU-4: 0.378
CIDEr: 126.7
CHAIR_i: 0.052
CHAIR_s: 0.121
```

### Evaluate Medical Domain

```bash
python Medical_Domain/evaluate_medical.py \
    --checkpoint ./checkpoints/medical/best_rl_model.pth \
    --data_dir ./data/MIMIC-CXR \
    --output_dir ./results/medical \
    --dataset mimic_cxr \
    --split test \
    --beam_size 3 \
    --use_radgraph
```

**Kết quả mong đợi:**
```
BLEU-4: 0.153
CIDEr: 39.8
RadGraph F1: 0.361
```

## 🎨 Bước 5: Inference (Demo)

### Python Script

```python
from demo_inference import ImageCaptioner

# General domain
captioner = ImageCaptioner(
    checkpoint_path='./checkpoints/general/best_rl_model.pth',
    domain='general',
    device='cuda'
)

caption = captioner.generate_caption('path/to/image.jpg', beam_size=3)
print(f"Caption: {caption}")

# Visualize
captioner.visualize('path/to/image.jpg')
```

### Medical domain

```python
medical_captioner = ImageCaptioner(
    checkpoint_path='./checkpoints/medical/best_rl_model.pth',
    domain='medical',
    device='cuda'
)

report = medical_captioner.generate_caption('path/to/xray.dcm', beam_size=3)
print(f"Report: {report}")
```

## 🔧 Bước 6: Troubleshooting

### CUDA Out of Memory

```bash
# Giảm batch size
--batch_size 8

# Giảm số regions
--num_regions 20

# Use gradient accumulation (sửa trong trainer.py)
```

### Slow Training

```bash
# Sử dụng mixed precision training (thêm vào trainer)
from torch.cuda.amp import autocast, GradScaler

# Tăng num_workers
--num_workers 8

# Pre-extract features (tạo file .h5)
```

### RadGraph/CheXbert không hoạt động

```bash
# Install RadGraph dependencies
pip install radgraph
python -m spacy download en_core_sci_sm

# Skip medical metrics nếu không cần
--use_radgraph False
--use_chexbert False
```

## 📊 Bước 7: Monitoring Training

### TensorBoard

```bash
# Thêm vào training scripts:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs')

# Xem training progress
tensorboard --logdir=./logs
```

### Weights & Biases (Optional)

```bash
# Install wandb
pip install wandb

# Login
wandb login

# Enable trong config
use_wandb: true
wandb_project: "my-captioning-project"
```

## 🎯 Bước 8: Thử nghiệm với các cấu hình khác

### Ablation Study: Chỉ dùng CIDEr reward

```bash
python General_Domain/train_general.py \
    --reward_cider_weight 1.0 \
    --reward_chair_weight 0.0 \
    --skip_xe
```

### Thử nghiệm với model nhỏ hơn

```bash
python General_Domain/train_general.py \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 4 \
    --d_ff 1024
```

### Thử nghiệm với beam search khác nhau

```bash
# Greedy (beam_size=1)
python General_Domain/evaluate_general.py --beam_size 1

# Beam search (beam_size=5)
python General_Domain/evaluate_general.py --beam_size 5
```

## 📚 Tài liệu tham khảo

- **Full Documentation**: Xem `README.md`
- **Module Documentation**: Mỗi file Python có docstrings chi tiết
- **Config Examples**: Xem `configs/` folder
- **Demo Code**: `demo_inference.py`

## 💡 Tips & Tricks

### Tăng tốc Training

1. **Pre-extract region features**: Chạy Faster R-CNN một lần, lưu features vào HDF5
2. **Mixed Precision Training**: Sử dụng `torch.cuda.amp`
3. **Gradient Accumulation**: Tăng effective batch size
4. **Multi-GPU Training**: Sử dụng `DistributedDataParallel`

### Cải thiện Performance

1. **Data Augmentation**: Thêm random crops, flips
2. **Curriculum Learning**: Train trên captions ngắn trước
3. **Ensemble**: Kết hợp nhiều models
4. **Post-processing**: Spell checking, grammar correction

### Debug

```bash
# Test với subset nhỏ
head -n 1000 data.json > data_small.json

# In ra generated captions định kỳ
--log_every_n_steps 10

# Visualize attention weights
--visualize_attention True
```

## ✅ Checklist

- [ ] Environment setup hoàn tất
- [ ] Data downloaded và preprocessed
- [ ] XE training completed (Phase 1)
- [ ] RL training completed (Phase 2)
- [ ] Evaluation chạy thành công
- [ ] Metrics đạt baseline
- [ ] Demo inference hoạt động
- [ ] Documentation đọc và hiểu

## 🆘 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra lại requirements
2. Xem logs trong `./logs/`
3. Đọc error messages cẩn thận
4. Tạo issue với thông tin chi tiết:
   - Python version
   - CUDA version
   - Error traceback
   - Config sử dụng

Good luck! 🚀
