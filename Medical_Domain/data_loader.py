"""
Data Loaders for Medical Domain (MIMIC-CXR, VinDr-CXR, IU X-Ray)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import json
import os
import h5py
import numpy as np
import pydicom
from typing import Dict, List, Tuple, Optional


class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR Dataset for Chest X-ray Report Generation
    """
    
    def __init__(
        self,
        image_dir: str,
        split_file: str,
        report_file: str,
        split: str = 'train',
        transform=None,
        max_seq_len: int = 200,
        vocab: Dict[str, int] = None,
        build_vocab: bool = False,
        use_findings_only: bool = True
    ):
        """
        Args:
            image_dir: Directory containing MIMIC-CXR DICOM files
            split_file: CSV file with train/val/test splits
            report_file: CSV file with reports
            split: 'train', 'validate', or 'test'
            transform: Image transforms
            max_seq_len: Maximum report length
            vocab: Vocabulary dictionary
            build_vocab: Whether to build vocabulary
            use_findings_only: Use only findings section (vs full report)
        """
        self.image_dir = image_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.use_findings_only = use_findings_only
        
        # Load split information
        split_df = pd.read_csv(split_file)
        self.data = split_df[split_df['split'] == split].reset_index(drop=True)
        
        # Load reports
        reports_df = pd.read_csv(report_file)
        self.reports = {}
        
        for _, row in reports_df.iterrows():
            study_id = row['study_id']
            if use_findings_only:
                report = row['findings'] if pd.notna(row['findings']) else ""
            else:
                report = row['impression'] if pd.notna(row['impression']) else ""
            
            self.reports[study_id] = report
        
        # Build or load vocabulary
        if build_vocab:
            self.vocab = self._build_vocab()
        else:
            assert vocab is not None, "Must provide vocab if not building"
            self.vocab = vocab
        
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        # Image transforms (medical-specific)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],  # Different normalization for X-rays
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transform
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from all reports."""
        word_freq = {}
        
        for study_id in self.data['study_id']:
            if study_id in self.reports:
                report = self.reports[study_id].lower()
                tokens = report.split()
                
                for word in tokens:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create vocabulary (keep words with freq >= 3 for medical domain)
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
            if freq >= 3:
                vocab[word] = len(vocab)
        
        print(f"Medical vocabulary size: {len(vocab)}")
        return vocab
    
    def _load_dicom_image(self, dicom_path: str) -> Image.Image:
        """Load and preprocess DICOM image."""
        try:
            dcm = pydicom.dcmread(dicom_path)
            image_array = dcm.pixel_array
            
            # Normalize to 0-255
            image_array = image_array.astype(np.float32)
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            image_array = (image_array * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image_array)
            
            return image
        except:
            # Return black image if loading fails
            return Image.new('L', (224, 224))
    
    def _tokenize_report(self, report: str) -> List[int]:
        """Convert report to token IDs."""
        tokens = report.lower().split()
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
        row = self.data.iloc[idx]
        
        study_id = row['study_id']
        subject_id = row['subject_id']
        dicom_id = row['dicom_id']
        
        # Construct DICOM path
        # MIMIC-CXR structure: p{subject_id}/s{study_id}/{dicom_id}.dcm
        subject_dir = f"p{str(subject_id)[:2]}/p{subject_id}"
        study_dir = f"s{study_id}"
        dicom_path = os.path.join(self.image_dir, subject_dir, study_dir, f"{dicom_id}.dcm")
        
        # Load image
        image = self._load_dicom_image(dicom_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Get report
        report = self.reports.get(study_id, "")
        report_ids = self._tokenize_report(report)
        
        return {
            'image': image,
            'image_id': study_id,
            'captions': torch.tensor(report_ids, dtype=torch.long),
            'captions_text': report,
            'subject_id': subject_id,
            'dicom_id': dicom_id
        }


class VinDrCXRDataset(Dataset):
    """
    VinDr-CXR Dataset
    Contains bounding boxes for 14 thoracic pathologies
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        split: str = 'train',
        transform=None,
        max_boxes: int = 20
    ):
        """
        Args:
            image_dir: Directory containing VinDr-CXR images
            annotation_file: CSV file with pathology annotations
            split: 'train' or 'test'
            transform: Image transforms
            max_boxes: Maximum number of bounding boxes
        """
        self.image_dir = image_dir
        self.split = split
        self.max_boxes = max_boxes
        
        # Load annotations
        annotations_df = pd.read_csv(annotation_file)
        
        # Group by image_id
        self.data = {}
        for _, row in annotations_df.iterrows():
            image_id = row['image_id']
            
            if image_id not in self.data:
                self.data[image_id] = {
                    'boxes': [],
                    'labels': [],
                    'image_id': image_id
                }
            
            # Add bounding box
            if pd.notna(row['x_min']):  # Has bounding box
                box = [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
                self.data[image_id]['boxes'].append(box)
                self.data[image_id]['labels'].append(row['class_name'])
        
        self.image_ids = list(self.data.keys())
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        data = self.data[image_id]
        
        # Load image (assuming PNG format)
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        try:
            image = Image.open(image_path).convert('L')
        except:
            image = Image.new('L', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Get boxes and labels
        boxes = data['boxes']
        labels = data['labels']
        
        # Pad or truncate
        if len(boxes) > self.max_boxes:
            boxes = boxes[:self.max_boxes]
            labels = labels[:self.max_boxes]
        else:
            padding_needed = self.max_boxes - len(boxes)
            boxes += [[0, 0, 0, 0]] * padding_needed
            labels += [''] * padding_needed
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return {
            'image': image,
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'num_boxes': len(data['boxes'])
        }


class IUXRayDataset(Dataset):
    """
    IU X-Ray Dataset for Medical Report Generation
    Smaller dataset used for cross-dataset evaluation
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        split: str = 'train',
        transform=None,
        max_seq_len: int = 200,
        vocab: Dict[str, int] = None
    ):
        """
        Args:
            image_dir: Directory containing IU X-Ray images
            annotation_file: JSON file with annotations
            split: 'train', 'val', or 'test'
            transform: Image transforms
            max_seq_len: Maximum report length
            vocab: Vocabulary dictionary
        """
        self.image_dir = image_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.data = [item for item in data if item['split'] == split]
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load images (IU X-Ray typically has 2 views per study)
        images = []
        for image_file in item['images']:
            image_path = os.path.join(self.image_dir, image_file)
            try:
                image = Image.open(image_path).convert('L')
            except:
                image = Image.new('L', (224, 224))
            
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
        
        # Use first image (or combine multiple views)
        image = images[0] if len(images) > 0 else torch.zeros(3, 224, 224)
        
        # Get report
        report = item['report']
        
        return {
            'image': image,
            'image_id': item['id'],
            'captions_text': report,
            'num_images': len(images)
        }


def get_medical_dataloader(
    dataset_name: str,
    data_dir: str,
    split: str = 'train',
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    vocab: Dict[str, int] = None
) -> DataLoader:
    """
    Factory function to create data loaders for medical domain.
    
    Args:
        dataset_name: 'mimic_cxr', 'vindr_cxr', or 'iu_xray'
        data_dir: Root directory for dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        vocab: Vocabulary dictionary
        
    Returns:
        DataLoader instance
    """
    if dataset_name == 'mimic_cxr':
        image_dir = os.path.join(data_dir, 'files')
        split_file = os.path.join(data_dir, 'mimic-cxr-2.0.0-split.csv')
        report_file = os.path.join(data_dir, 'mimic_cxr_sectioned.csv')
        
        dataset = MIMICCXRDataset(
            image_dir=image_dir,
            split_file=split_file,
            report_file=report_file,
            split=split,
            vocab=vocab,
            build_vocab=(split == 'train' and vocab is None)
        )
        
    elif dataset_name == 'vindr_cxr':
        image_dir = os.path.join(data_dir, 'images', split)
        annotation_file = os.path.join(data_dir, f'annotations_{split}.csv')
        
        dataset = VinDrCXRDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            split=split
        )
        
    elif dataset_name == 'iu_xray':
        image_dir = os.path.join(data_dir, 'images')
        annotation_file = os.path.join(data_dir, 'annotations.json')
        
        dataset = IUXRayDataset(
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
