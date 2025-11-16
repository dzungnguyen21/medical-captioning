"""
Region-based Feature Extractor using Faster R-CNN
Extracts bottom-up attention features for both General and Medical domains
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
from typing import List, Tuple, Dict
import numpy as np


class RegionFeatureExtractor(nn.Module):
    """
    Region-based feature extractor using Faster R-CNN.
    Returns region features and object detection results for grounding.
    """
    
    def __init__(
        self, 
        pretrained: bool = True,
        num_regions: int = 36,
        feature_dim: int = 2048,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Args:
            pretrained: Use pretrained Faster R-CNN
            num_regions: Maximum number of regions to extract
            feature_dim: Dimension of region features
            conf_threshold: Confidence threshold for object detection
            nms_threshold: NMS threshold for object detection
        """
        super().__init__()
        
        self.num_regions = num_regions
        self.feature_dim = feature_dim
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        
        # Load Faster R-CNN backbone
        self.detector = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        self.detector.eval()
        
        # Feature projection layer
        self.feature_proj = nn.Linear(1024, feature_dim)
        
        # Spatial encoding (normalized box coordinates)
        self.spatial_dim = 5  # [x1, y1, x2, y2, area]
        self.spatial_proj = nn.Linear(self.spatial_dim, feature_dim)
        
    def extract_features_from_boxes(
        self, 
        images: torch.Tensor, 
        boxes: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from given bounding boxes using RoI pooling.
        
        Args:
            images: Batch of images [B, 3, H, W]
            boxes: List of boxes for each image [N, 4]
            
        Returns:
            region_features: [B, num_regions, feature_dim]
            spatial_features: [B, num_regions, spatial_dim]
        """
        batch_size = images.size(0)
        
        # Extract backbone features
        with torch.no_grad():
            # Get intermediate features from backbone
            features = self.detector.backbone(images)
            # Use features from the last layer
            feature_map = features['pool']  # [B, 1024, H', W']
        
        region_features_list = []
        spatial_features_list = []
        
        for i in range(batch_size):
            if len(boxes[i]) == 0:
                # No boxes detected, use zero features
                region_feat = torch.zeros(self.num_regions, self.feature_dim).to(self.device)
                spatial_feat = torch.zeros(self.num_regions, self.spatial_dim).to(self.device)
            else:
                # RoI pooling
                roi_features = roi_align(
                    feature_map[i:i+1],
                    [boxes[i]],
                    output_size=(7, 7),
                    spatial_scale=1.0/32,  # Depends on feature map stride
                    aligned=True
                )
                
                # Average pooling
                roi_features = roi_features.mean(dim=[2, 3])  # [N, 1024]
                
                # Limit to num_regions
                if roi_features.size(0) > self.num_regions:
                    roi_features = roi_features[:self.num_regions]
                    current_boxes = boxes[i][:self.num_regions]
                else:
                    current_boxes = boxes[i]
                
                # Pad if needed
                if roi_features.size(0) < self.num_regions:
                    padding = torch.zeros(
                        self.num_regions - roi_features.size(0), 
                        1024
                    ).to(self.device)
                    roi_features = torch.cat([roi_features, padding], dim=0)
                    
                    box_padding = torch.zeros(
                        self.num_regions - current_boxes.size(0), 
                        4
                    ).to(self.device)
                    current_boxes = torch.cat([current_boxes, box_padding], dim=0)
                
                # Project features
                region_feat = self.feature_proj(roi_features)
                
                # Compute spatial features (normalized coordinates)
                img_h, img_w = images.size(2), images.size(3)
                spatial_feat = self._compute_spatial_features(
                    current_boxes, img_h, img_w
                )
            
            region_features_list.append(region_feat)
            spatial_features_list.append(spatial_feat)
        
        region_features = torch.stack(region_features_list, dim=0)  # [B, num_regions, feature_dim]
        spatial_features = torch.stack(spatial_features_list, dim=0)  # [B, num_regions, spatial_dim]
        
        return region_features, spatial_features
    
    def _compute_spatial_features(
        self, 
        boxes: torch.Tensor, 
        img_h: int, 
        img_w: int
    ) -> torch.Tensor:
        """
        Compute spatial features from bounding boxes.
        
        Args:
            boxes: [N, 4] in format [x1, y1, x2, y2]
            img_h, img_w: Image height and width
            
        Returns:
            spatial_features: [N, 5] normalized spatial features
        """
        # Normalize coordinates
        x1 = boxes[:, 0] / img_w
        y1 = boxes[:, 1] / img_h
        x2 = boxes[:, 2] / img_w
        y2 = boxes[:, 3] / img_h
        
        # Compute area
        area = (x2 - x1) * (y2 - y1)
        
        spatial_features = torch.stack([x1, y1, x2, y2, area], dim=1)
        
        return spatial_features
    
    def forward(
        self, 
        images: torch.Tensor,
        return_detections: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract region features.
        
        Args:
            images: Batch of images [B, 3, H, W]
            return_detections: Whether to return detection results
            
        Returns:
            Dictionary containing:
                - region_features: [B, num_regions, feature_dim]
                - spatial_features: [B, num_regions, spatial_dim]
                - boxes: List of detected boxes (if return_detections=True)
                - labels: List of detected labels (if return_detections=True)
                - scores: List of detection scores (if return_detections=True)
        """
        self.detector.eval()
        
        with torch.no_grad():
            # Get detections
            detections = self.detector(images)
        
        # Process detections
        boxes_list = []
        labels_list = []
        scores_list = []
        
        for det in detections:
            # Filter by confidence
            keep = det['scores'] > self.conf_threshold
            boxes = det['boxes'][keep]
            labels = det['labels'][keep]
            scores = det['scores'][keep]
            
            # Apply NMS
            keep_nms = torchvision.ops.nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep_nms]
            labels = labels[keep_nms]
            scores = scores[keep_nms]
            
            boxes_list.append(boxes)
            labels_list.append(labels)
            scores_list.append(scores)
        
        # Extract features from detected boxes
        region_features, spatial_features = self.extract_features_from_boxes(
            images, boxes_list
        )
        
        # Combine region and spatial features
        spatial_encoded = self.spatial_proj(spatial_features)
        combined_features = region_features + spatial_encoded
        
        output = {
            'region_features': combined_features,
            'spatial_features': spatial_features,
        }
        
        if return_detections:
            output['boxes'] = boxes_list
            output['labels'] = labels_list
            output['scores'] = scores_list
        
        return output


class MedicalRegionExtractor(RegionFeatureExtractor):
    """
    Medical-specific region extractor.
    Can be fine-tuned on medical datasets with pathology bounding boxes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # For medical images, we might want to add:
        # 1. Lung segmentation model
        # 2. Pathology detection model (trained on VinDr-CXR)
        # 3. Anatomical region detection
        
    def extract_anatomical_regions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract predefined anatomical regions from chest X-rays.
        This can use a simple grid or a trained segmentation model.
        """
        # Placeholder: divide image into anatomical regions
        # In practice, you would use a lung segmentation model
        batch_size = images.size(0)
        
        # Simple grid-based approach (for demonstration)
        # You should replace this with actual lung segmentation
        grid_regions = self._create_grid_regions(images)
        
        return grid_regions
    
    def _create_grid_regions(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Create a grid of regions for medical images.
        This is a simple baseline; you should use actual pathology detection.
        """
        batch_size = images.size(0)
        img_h, img_w = images.size(2), images.size(3)
        
        # Create a 3x3 grid
        grid_size = 3
        boxes_list = []
        
        for b in range(batch_size):
            boxes = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x1 = j * img_w / grid_size
                    y1 = i * img_h / grid_size
                    x2 = (j + 1) * img_w / grid_size
                    y2 = (i + 1) * img_h / grid_size
                    boxes.append([x1, y1, x2, y2])
            
            boxes_list.append(torch.tensor(boxes, dtype=torch.float32).to(self.device))
        
        return boxes_list
