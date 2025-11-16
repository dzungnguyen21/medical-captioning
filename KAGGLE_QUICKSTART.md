```markdown
# Hướng Dẫn Nhanh: Huấn Luyện Image Captioning Y Tế Trong 4 Ngày
## Tối ưu cho Kaggle T4 x2 GPU

---

## 📋 Tổng Quan

**Mục tiêu**: Xây dựng hệ thống Medical Image Captioning giảm thiểu hallucination trong 4 ngày với ràng buộc tài nguyên.

**Tài nguyên**:
- 2x Tesla T4 GPU (16GB VRAM mỗi card) trên Kaggle
- ~30 giờ GPU time / tuần

**Dataset**: IU X-Ray (~7,470 ảnh X-quang ngực)

**Chiến lược**: 
1. **Không train from scratch** - Sử dụng pre-trained VLM (Vision-Language Model)
2. **LoRA fine-tuning** - Chỉ train một phần nhỏ parameters
3. **Keyword-based reward** - Thay vì RadGraph phức tạp

---

## 🗓️ Lịch Trình 4 Ngày

### **Ngày 1 (8 giờ): Baseline - Cross-Entropy Training**

**Mục tiêu**: Có một model baseline hoạt động tốt

#### Bước 1: Setup môi trường trên Kaggle

```bash
# Trên Kaggle Notebook
# Settings -> Accelerator -> GPU T4 x2
# Settings -> Internet -> ON

# Clone repository
!git clone https://github.com/your-repo/medical-captioning.git
%cd medical-captioning

# Install dependencies
!pip install transformers peft bitsandbytes -q
!pip install pycocoevalcap nltk -q
!pip install accelerate einops -q
```

#### Bước 2: Download IU X-Ray dataset

```bash
# Option 1: From Kaggle Dataset
# Add "IU X-Ray" dataset vào notebook (nếu có sẵn)

# Option 2: Download từ OpenI
!wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
!tar -xzf NLMCXR_png.tgz -C data/

# Cấu trúc mong đợi:
# data/IU_XRAY/
#   ├── images/
#   └── indiana_reports.csv
```

#### Bước 3: Chạy training Ngày 1

```bash
# Sử dụng ViT-GPT2 (nhanh, nhẹ)
python 4Day_Scripts/day1_baseline.py \
    --data_dir data/IU_XRAY \
    --model_type vit-gpt2 \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    --use_lora \
    --use_fp16 \
    --use_multi_gpu

# Hoặc sử dụng BLIP-2 (tốt hơn nhưng chậm hơn)
python 4Day_Scripts/day1_baseline.py \
    --data_dir data/IU_XRAY \
    --model_type blip2 \
    --epochs 10 \
    --batch_size 8 \
    --lr 5e-5 \
    --use_8bit \
    --use_lora
```

**Thời gian dự kiến**: 4-6 giờ

**Kết quả mong đợi**:
- BLEU-4: ~0.15-0.20
- CIDEr: ~0.30-0.50
- Checkpoint saved: `checkpoints/day1_baseline/best_model.pt`

#### Troubleshooting Ngày 1

**Lỗi OOM (Out of Memory)**:
```bash
# Giảm batch size
--batch_size 8

# Hoặc giảm sequence length
--max_length 64
```

**Training quá chậm**:
```bash
# Giảm số epochs
--epochs 5

# Hoặc dùng ViT-GPT2 thay vì BLIP-2
--model_type vit-gpt2
```

---

### **Ngày 2 (8 giờ): SCST - Reinforcement Learning**

**Mục tiêu**: Cải thiện factual accuracy, giảm hallucination

#### Bước 1: Kiểm tra checkpoint từ Ngày 1

```bash
# Verify checkpoint exists
ls -lh checkpoints/day1_baseline/best_model.pt
```

#### Bước 2: Chạy SCST training

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

**⚠️ Lưu ý quan trọng cho RL**:

1. **Learning rate phải rất nhỏ** (1e-6, không lớn hơn 5e-6)
2. **Batch size nhỏ hơn** (8 thay vì 16) vì RL tốn bộ nhớ hơn
3. **Số epochs ít hơn** (5 thay vì 10) vì RL dễ overfit
4. **Monitor rewards**: Nếu reward giảm liên tục, STOP ngay!

**Thời gian dự kiến**: 6-8 giờ

**Kết quả mong đợi**:
- CIDEr improvement: +0.05-0.15
- Keyword F1 improvement: +3-8%
- Hallucination rate reduction: -5-10%

#### Kiểm tra kết quả SCST

```python
# So sánh Baseline vs SCST
import json

with open("checkpoints/day1_baseline/history.json") as f:
    baseline = json.load(f)

with open("checkpoints/day2_scst/history.json") as f:
    scst = json.load(f)

print(f"Baseline CIDEr: {baseline['best_cider']:.4f}")
print(f"SCST CIDEr: {scst['best_cider']:.4f}")
print(f"Improvement: {scst['best_cider'] - baseline['best_cider']:+.4f}")
```

#### RL Troubleshooting

**Rewards trở nên rất âm** (< -1.0):
```bash
# RL đang diverge, giảm learning rate
--lr 5e-7

# Hoặc stop training và dùng checkpoint tốt nhất từ Ngày 1
```

**Training quá chậm**:
```bash
# Giảm epochs
--epochs 3

# Tăng batch size một chút (nếu VRAM còn dư)
--batch_size 12
```

---

### **Ngày 3 (4 giờ): Ensemble và Analysis**

**Mục tiêu**: Kết hợp nhiều models để tăng performance

#### Bước 1: Ensemble evaluation

```bash
python 4Day_Scripts/day3_ensemble.py \
    --data_dir data/IU_XRAY \
    --model_type vit-gpt2 \
    --checkpoints \
        checkpoints/day1_baseline/best_model.pt \
        checkpoints/day2_scst/best_model.pt \
    --model_names baseline scst \
    --ensemble_method voting \
    --evaluate_test
```

**Thời gian dự kiến**: 2-3 giờ

**Kết quả mong đợi**:
- Ensemble thường cải thiện +0.02-0.05 CIDEr so với single model

#### Bước 2: Thử nghiệm (nếu còn thời gian)

Nếu còn thời gian, có thể thử:

**Option A**: Train thêm một model với hyperparameter khác
```bash
python 4Day_Scripts/day1_baseline.py \
    --epochs 8 \
    --lr 3e-5 \
    --checkpoint_dir checkpoints/day3_variant
```

**Option B**: Fine-tune thêm với hybrid loss
```python
# Kết hợp CE + SCST reward trong một epoch
# (code advanced - xem trainer.py)
```

---

### **Ngày 4 (4 giờ): Final Evaluation và Report**

**Mục tiêu**: Đánh giá toàn diện và viết báo cáo

#### Bước 1: Comprehensive evaluation

```bash
python 4Day_Scripts/day4_evaluate.py \
    --data_dir data/IU_XRAY \
    --model_type vit-gpt2 \
    --checkpoints \
        checkpoints/day1_baseline/best_model.pt \
        checkpoints/day2_scst/best_model.pt \
    --model_names baseline scst \
    --output_dir results/final
```

**Output**:
- `results/final/FINAL_REPORT.md`: Báo cáo chi tiết
- `results/final/evaluation_results.json`: Metrics số liệu
- Best/worst examples cho qualitative analysis

**Thời gian dự kiến**: 1-2 giờ

#### Bước 2: Visualize results

```python
import json
import matplotlib.pyplot as plt

# Load results
with open("results/final/evaluation_results.json") as f:
    results = json.load(f)

# Plot comparison
models = list(results.keys())
ciders = [results[m]["CIDEr"] for m in models]
kw_f1s = [results[m]["Keyword_F1"] for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(models, ciders)
ax1.set_ylabel("CIDEr")
ax1.set_title("CIDEr Comparison")

ax2.bar(models, kw_f1s)
ax2.set_ylabel("Keyword F1")
ax2.set_title("Keyword F1 Comparison")

plt.tight_layout()
plt.savefig("results/final/comparison.png", dpi=150)
```

#### Bước 3: Viết báo cáo (2 giờ)

1. **Quantitative results**: Bảng so sánh metrics
2. **Qualitative analysis**: Best/worst examples
3. **Discussion**: 
   - SCST có giúp ích không?
   - Hallucination có giảm không?
   - Keyword matching có hợp lý không?
4. **Conclusion**: Lessons learned

---

## 📊 Expected Results (Tham khảo)

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr | Keyword F1 | Hallucination |
|-------|--------|--------|---------|-------|------------|---------------|
| Baseline (CE) | 0.18 | 0.24 | 0.42 | 0.45 | 0.62 | 0.28 |
| SCST (RL) | 0.20 | 0.26 | 0.44 | 0.52 | 0.68 | 0.21 |
| Ensemble | 0.21 | 0.27 | 0.45 | 0.54 | 0.70 | 0.19 |

**Improvement từ Baseline → SCST**:
- CIDEr: +15%
- Keyword F1: +10%
- Hallucination: -25%

---

## 💡 Tips Quan Trọng Cho Kaggle

### 1. Quản lý GPU Time

Kaggle cho ~30 giờ GPU/tuần. **Lưu checkpoint thường xuyên!**

```bash
# Auto-save mỗi epoch
--eval_every 1

# Checkpoint directory persistent
!mkdir -p /kaggle/working/checkpoints
!cp -r checkpoints/* /kaggle/working/
```

### 2. Tăng tốc Training

```bash
# Mixed precision (BẮT BUỘC)
--use_fp16

# Multi-GPU
--use_multi_gpu

# DataLoader workers
--num_workers 4
```

### 3. Giảm VRAM Usage

```bash
# ViT-GPT2 (nhẹ nhất)
--model_type vit-gpt2

# 8-bit quantization (BLIP-2)
--use_8bit

# Gradient accumulation (nếu batch size quá nhỏ)
--gradient_accumulation_steps 2
```

### 4. Debug Nhanh

```bash
# Test với 1 epoch đầu
--epochs 1

# Test với subset nhỏ
--max_samples 100
```

---

## 🚨 Common Issues & Solutions

### Issue 1: "CUDA Out of Memory"

**Solution**:
```bash
# Giảm batch size
--batch_size 4

# Giảm sequence length
--max_length 64

# Sử dụng gradient checkpointing (trong trainer.py)
```

### Issue 2: "RL rewards dropping"

**Solution**:
```bash
# STOP training ngay!
# Giảm learning rate
--lr 5e-7

# Hoặc quay lại baseline checkpoint
```

### Issue 3: "Training quá chậm"

**Solution**:
```bash
# Dùng ViT-GPT2 thay vì BLIP-2
--model_type vit-gpt2

# Giảm num_workers nếu CPU bottleneck
--num_workers 2

# Giảm eval frequency
--eval_every 5
```

### Issue 4: "IU X-Ray data not found"

**Solution**:
```bash
# Download manual từ OpenI
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz

# Hoặc dùng Kaggle dataset (search "IU X-Ray")
# Add dataset to notebook
```

---

## 📁 Cấu Trúc Project

```
medical_img_captioning_train/
├── Fast_Models/
│   ├── blip2_wrapper.py          # BLIP-2 with LoRA/8-bit
│   └── vit_gpt2_wrapper.py       # ViT-GPT2 (faster)
├── Fast_Data/
│   └── iu_xray_loader.py         # IU X-Ray dataset
├── Fast_Rewards/
│   └── keyword_reward.py         # Keyword-based reward
├── Fast_Training/
│   └── trainer.py                # CE + SCST trainer
├── 4Day_Scripts/
│   ├── day1_baseline.py          # Day 1: CE training
│   ├── day2_scst.py              # Day 2: RL training
│   ├── day3_ensemble.py          # Day 3: Ensemble
│   └── day4_evaluate.py          # Day 4: Evaluation
└── KAGGLE_QUICKSTART.md          # This file
```

---

## ✅ Checklist

### Before Starting:
- [ ] Kaggle account với GPU quota
- [ ] IU X-Ray dataset downloaded
- [ ] Dependencies installed
- [ ] GPU T4 x2 activated

### Day 1:
- [ ] Baseline training complete (4-6 giờ)
- [ ] Checkpoint saved: `day1_baseline/best_model.pt`
- [ ] CIDEr ~0.3-0.5

### Day 2:
- [ ] SCST training complete (6-8 giờ)
- [ ] Rewards improving (not dropping)
- [ ] CIDEr improvement +0.05-0.15

### Day 3:
- [ ] Ensemble evaluation done
- [ ] Best model selected

### Day 4:
- [ ] Final evaluation complete
- [ ] Report written
- [ ] Results saved to `results/final/`

---

## 🎯 Success Criteria

**Minimum viable**:
- BLEU-4 > 0.15
- CIDEr > 0.40
- Keyword F1 > 0.60
- Hallucination < 0.30

**Good result**:
- BLEU-4 > 0.20
- CIDEr > 0.50
- Keyword F1 > 0.65
- Hallucination < 0.25

**Excellent result**:
- BLEU-4 > 0.25
- CIDEr > 0.60
- Keyword F1 > 0.70
- Hallucination < 0.20

---

## 📚 References

- **IU X-Ray**: Demner-Fushman et al., 2016
- **BLIP-2**: Li et al., 2023
- **SCST**: Rennie et al., 2017
- **LoRA**: Hu et al., 2021

---

## 🆘 Support

Nếu gặp vấn đề:
1. Check `TROUBLESHOOTING.md`
2. Review error messages carefully
3. Search Kaggle discussions
4. Check GPU usage: `nvidia-smi`

---

**Good luck! 🚀**

Nhớ rằng: **4 ngày là rất ngắn**. Tập trung vào làm cho code chạy được trước, tối ưu sau!
```
