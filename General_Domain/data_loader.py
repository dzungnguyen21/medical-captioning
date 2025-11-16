"""
Data Loaders for General Domain (MS-COCO, Visual Genome)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle


class COCOCaptionDataset(Dataset):
    """
    MS-COCO Caption Dataset (Karpathy Split)
    """
    
    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        split: str = 'train',
        transform=None,
        max_seq_len: int = 50,
        vocab: Dict[str, int] = None,
        build_vocab: bool = False
    ):
        """
        Args:
            image_dir: Directory containing COCO images
            caption_file: Path to Karpathy split JSON file
            split: 'train', 'val', or 'test'
            transform: Image transforms
            max_seq_len: Maximum caption length
            vocab: Vocabulary dictionary (word -> idx)
            build_vocab: Whether to build vocabulary from captions
        """
        self.image_dir = image_dir
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load captions
        with open(caption_file, 'r') as f:
            data = json.load(f)
        
        # Filter by split
        self.data = [item for item in data['images'] if item['split'] == split]
        
        # Build or load vocabulary
        if build_vocab:
            self.vocab = self._build_vocab()
        else:
            assert vocab is not None, "Must provide vocab if not building"
            self.vocab = vocab
        
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from all captions."""
        word_freq = {}
        
        for item in self.data:
            for caption in item['sentences']:
                tokens = caption['tokens']
                for word in tokens:
                    word = word.lower()
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create vocabulary (keep words with freq >= 5)
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
            if freq >= 5:
                vocab[word] = len(vocab)
        
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def _tokenize_caption(self, caption: str) -> List[int]:
        """Convert caption to token IDs."""
        tokens = caption.lower().split()
        token_ids = [self.vocab.get('<START>')]
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab.get('<UNK>')))
        
        token_ids.append(self.vocab.get('<END>'))
        
        # Pad or truncate
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids += [self.vocab.get('<PAD>')] * (self.max_seq_len - len(token_ids))
        
        return token_ids
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load image
        image_file = item['filename']
        if 'coco' in self.image_dir.lower():
            # COCO format: train2014/val2014
            if 'train' in item['filepath']:
                image_path = os.path.join(self.image_dir, 'train2014', image_file)
            else:
                image_path = os.path.join(self.image_dir, 'val2014', image_file)
        else:
            image_path = os.path.join(self.image_dir, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Fallback to black image if file not found
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Get captions (use first caption for training, all for evaluation)
        if self.split == 'train':
            caption = item['sentences'][0]['raw']
            caption_ids = self._tokenize_caption(caption)
        else:
            # For validation/test, return all captions
            captions = [sent['raw'] for sent in item['sentences']]
            caption = captions[0]  # Use first for encoding
            caption_ids = self._tokenize_caption(caption)
        
        return {
            'image': image,
            'image_id': item['cocoid'],
            'captions': torch.tensor(caption_ids, dtype=torch.long),
            'captions_text': caption if self.split == 'train' else captions,
            'filename': image_file
        }


class VisualGenomeDataset(Dataset):
    """
    Visual Genome Dataset for Region Pre-training
    Provides region-level features and dense captions
    """
    
    def __init__(
        self,
        image_dir: str,
        region_file: str,
        region_caption_file: str,
        transform=None,
        max_regions: int = 36,
        max_seq_len: int = 50,
        vocab: Dict[str, int] = None
    ):
        """
        Args:
            image_dir: Directory containing VG images
            region_file: HDF5 file with pre-extracted region features
            region_caption_file: JSON file with region descriptions
            transform: Image transforms
            max_regions: Maximum number of regions per image
            max_seq_len: Maximum caption length
            vocab: Vocabulary dictionary
        """
        self.image_dir = image_dir
        self.max_regions = max_regions
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        
        # Load region features (pre-extracted using Faster R-CNN)
        if os.path.exists(region_file):
            self.region_features = h5py.File(region_file, 'r')
        else:
            self.region_features = None
            print("Warning: Region features file not found")
        
        # Load region captions
        with open(region_caption_file, 'r') as f:
            self.region_captions = json.load(f)
        
        # Image IDs
        self.image_ids = list(self.region_captions.keys())
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Load region features (if available)
        if self.region_features:
            region_feats = self.region_features[str(image_id)][:]
            
            # Pad or truncate
            if region_feats.shape[0] > self.max_regions:
                region_feats = region_feats[:self.max_regions]
            else:
                padding = np.zeros((self.max_regions - region_feats.shape[0], region_feats.shape[1]))
                region_feats = np.vstack([region_feats, padding])
            
            region_feats = torch.tensor(region_feats, dtype=torch.float32)
        else:
            region_feats = torch.zeros(self.max_regions, 2048)
        
        # Get region captions (dense captions)
        region_caps = self.region_captions[image_id]['regions']
        
        # For simplicity, concatenate all region captions
        all_captions = [r['phrase'] for r in region_caps]
        combined_caption = '. '.join(all_captions[:5])  # Use first 5 region captions
        
        return {
            'image': image,
            'image_id': image_id,
            'region_features': region_feats,
            'captions_text': combined_caption,
            'num_regions': min(len(region_caps), self.max_regions)
        }


class NoCapsDataset(Dataset):
    """
    NoCaps Dataset for Hallucination Evaluation
    Contains novel objects not in COCO training set
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        split: str = 'val',
        transform=None,
        vocab: Dict[str, int] = None
    ):
        """
        Args:
            image_dir: Directory containing NoCaps images
            annotation_file: Path to annotations JSON
            split: 'val' or 'test'
            transform: Image transforms
            vocab: Vocabulary dictionary
        """
        self.image_dir = image_dir
        self.split = split
        self.vocab = vocab
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images = {img['id']: img for img in data['images']}
        self.annotations = {}
        
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann['caption'])
        
        self.image_ids = list(self.annotations.keys())
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Get captions
        captions = self.annotations[image_id]
        
        return {
            'image': image,
            'image_id': image_id,
            'captions_text': captions,
            'filename': image_info['file_name']
        }


def get_general_dataloader(
    dataset_name: str,
    data_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    vocab: Dict[str, int] = None
) -> DataLoader:
    """
    Factory function to create data loaders for general domain.
    
    Args:
        dataset_name: 'coco', 'visual_genome', or 'nocaps'
        data_dir: Root directory for dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        vocab: Vocabulary dictionary
        
    Returns:
        DataLoader instance
    """
    if dataset_name == 'coco':
        image_dir = os.path.join(data_dir, 'images')
        caption_file = os.path.join(data_dir, 'dataset_coco.json')
        
        dataset = COCOCaptionDataset(
            image_dir=image_dir,
            caption_file=caption_file,
            split=split,
            vocab=vocab,
            build_vocab=(split == 'train' and vocab is None)
        )
        
    elif dataset_name == 'visual_genome':
        image_dir = os.path.join(data_dir, 'images')
        region_file = os.path.join(data_dir, 'region_features.h5')
        region_caption_file = os.path.join(data_dir, 'region_descriptions.json')
        
        dataset = VisualGenomeDataset(
            image_dir=image_dir,
            region_file=region_file,
            region_caption_file=region_caption_file,
            vocab=vocab
        )
        
    elif dataset_name == 'nocaps':
        image_dir = os.path.join(data_dir, 'images')
        annotation_file = os.path.join(data_dir, f'nocaps_{split}_4500_captions.json')
        
        dataset = NoCapsDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            split=split,
            vocab=vocab
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
