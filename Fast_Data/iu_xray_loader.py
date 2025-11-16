"""
IU X-Ray Dataset Loader - Optimized for Fast Training
Indiana University Chest X-Ray dataset with ~7,470 images

Key Features:
- Fast preprocessing with caching
- Memory-efficient loading
- Support for both BLIP-2 and ViT-GPT2 formats
- Medical report parsing
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, ViTImageProcessor, AutoTokenizer


class IUXRayDataset(Dataset):
    """
    IU X-Ray Dataset
    
    Dataset Structure:
    data/IU_XRAY/
        ├── images/
        │   ├── CXR1_1_IM-0001-3001.png
        │   ├── CXR1_1_IM-0001-4001.png
        │   └── ...
        ├── indiana_reports.csv  (or .json)
        └── indiana_projections.csv
    
    CSV Format:
        - uid: Unique patient ID
        - projection: Frontal/Lateral
        - filename: Image filename
        - findings: Findings section
        - impression: Impression section
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # train, val, test
        image_processor = None,
        tokenizer = None,
        max_length: int = 128,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cache = use_cache
        
        if cache_dir is None:
            cache_dir = self.data_dir / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load annotations
        self._load_annotations()
        
        # Split data
        self._split_data()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_annotations(self):
        """Load and parse IU X-Ray annotations"""
        
        # Try to load from JSON first
        json_path = self.data_dir / "indiana_reports.json"
        csv_path = self.data_dir / "indiana_reports.csv"
        
        if json_path.exists():
            print(f"Loading annotations from {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Convert to list of samples
            self.all_samples = []
            for item in data:
                # Combine findings and impression
                report = ""
                if "findings" in item and item["findings"]:
                    report += item["findings"].strip() + " "
                if "impression" in item and item["impression"]:
                    report += item["impression"].strip()
                
                report = report.strip()
                
                # Skip empty reports
                if not report or report.lower() in [".", "none", "normal"]:
                    continue
                
                self.all_samples.append({
                    "image_path": self.data_dir / "images" / item["filename"],
                    "caption": self._clean_report(report),
                    "uid": item.get("uid", item["filename"])
                })
        
        elif csv_path.exists():
            print(f"Loading annotations from {csv_path}")
            df = pd.read_csv(csv_path)
            
            self.all_samples = []
            for idx, row in df.iterrows():
                # Combine findings and impression
                report = ""
                if pd.notna(row.get("findings")):
                    report += str(row["findings"]).strip() + " "
                if pd.notna(row.get("impression")):
                    report += str(row["impression"]).strip()
                
                report = report.strip()
                
                # Skip empty reports
                if not report or report.lower() in [".", "none", "normal"]:
                    continue
                
                image_path = self.data_dir / "images" / row["filename"]
                if not image_path.exists():
                    continue
                
                self.all_samples.append({
                    "image_path": image_path,
                    "caption": self._clean_report(report),
                    "uid": row.get("uid", row["filename"])
                })
        
        else:
            raise FileNotFoundError(
                f"Could not find annotations at {json_path} or {csv_path}\n"
                f"Please download IU X-Ray dataset and place it in {self.data_dir}"
            )
        
        print(f"Loaded {len(self.all_samples)} total samples")
    
    def _clean_report(self, text: str) -> str:
        """Clean and normalize medical report text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common artifacts
        text = text.replace("XXXX", "")
        text = text.replace("xxxx", "")
        
        # Lowercase (optional - medical terms are case-sensitive, but helps with vocab)
        # text = text.lower()
        
        # Limit length
        words = text.split()
        if len(words) > 100:  # Reasonable limit for X-ray reports
            text = " ".join(words[:100])
        
        return text.strip()
    
    def _split_data(self):
        """Split data into train/val/test"""
        
        # Use 70/15/15 split based on patient UID
        unique_uids = sorted(set(s["uid"] for s in self.all_samples))
        np.random.seed(42)
        np.random.shuffle(unique_uids)
        
        n_train = int(0.7 * len(unique_uids))
        n_val = int(0.15 * len(unique_uids))
        
        train_uids = set(unique_uids[:n_train])
        val_uids = set(unique_uids[n_train:n_train + n_val])
        test_uids = set(unique_uids[n_train + n_val:])
        
        if self.split == "train":
            uid_set = train_uids
        elif self.split == "val":
            uid_set = val_uids
        else:  # test
            uid_set = test_uids
        
        self.samples = [s for s in self.all_samples if s["uid"] in uid_set]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Process image
        if self.image_processor is not None:
            if isinstance(self.image_processor, Blip2Processor):
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values.squeeze(0)
            elif isinstance(self.image_processor, ViTImageProcessor):
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values.squeeze(0)
            else:
                raise ValueError("Unknown image processor type")
        else:
            # Default: resize and normalize
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            pixel_values = transform(image)
        
        # Tokenize caption
        caption = sample["caption"]
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = encoding.input_ids.squeeze(0)
            # Set padding tokens to -100 for loss computation
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = caption  # Return raw text
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "caption": caption,  # Keep original text for evaluation
            "image_path": str(sample["image_path"])
        }


def get_iu_xray_dataloaders(
    data_dir: str,
    model_type: str = "vit-gpt2",  # "vit-gpt2" or "blip2"
    batch_size: int = 16,
    num_workers: int = 4,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders for IU X-Ray
    
    Args:
        data_dir: Path to IU X-Ray dataset
        model_type: "vit-gpt2" or "blip2"
        batch_size: Batch size
        num_workers: Number of dataloader workers
        max_length: Maximum sequence length
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Load appropriate processor and tokenizer
    if model_type == "vit-gpt2":
        image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    elif model_type == "blip2":
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        image_processor = processor
        tokenizer = processor.tokenizer
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create datasets
    train_dataset = IUXRayDataset(
        data_dir=data_dir,
        split="train",
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = IUXRayDataset(
        data_dir=data_dir,
        split="val",
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = IUXRayDataset(
        data_dir=data_dir,
        split="test",
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader Statistics:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def test_iu_xray_loader():
    """Test IU X-Ray dataset loader"""
    print("\n" + "="*80)
    print("Testing IU X-Ray Dataset Loader")
    print("="*80)
    
    # You need to download IU X-Ray first and set the path
    data_dir = "data/IU_XRAY"
    
    if not os.path.exists(data_dir):
        print(f"\n⚠ Dataset not found at {data_dir}")
        print("Please download IU X-Ray dataset first:")
        print("  1. Download from: https://openi.nlm.nih.gov/")
        print("  2. Extract to: data/IU_XRAY/")
        print("  3. Ensure structure: data/IU_XRAY/images/ and indiana_reports.csv")
        return
    
    try:
        # Test ViT-GPT2 format
        print("\n[1] Testing ViT-GPT2 format...")
        train_loader, val_loader, test_loader = get_iu_xray_dataloaders(
            data_dir=data_dir,
            model_type="vit-gpt2",
            batch_size=4,
            num_workers=0
        )
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Sample caption: {batch['caption'][0][:100]}...")
        
        print("\n✓ IU X-Ray loader test passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Make sure IU X-Ray dataset is properly formatted")
    
    print("="*80)


if __name__ == "__main__":
    test_iu_xray_loader()
