"""
Demo Notebook: Quick Start Guide
Demonstrates how to use the trained models for inference
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Add modules to path
sys.path.append('..')

from Shared_Modules.region_encoder import RegionFeatureExtractor, MedicalRegionExtractor
from Shared_Modules.transformer_decoder import RegionAwareTransformerDecoder
from Shared_Modules.trainer import CaptioningModel


class ImageCaptioner:
    """Simple wrapper for caption generation."""
    
    def __init__(self, checkpoint_path, domain='general', device='cuda'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            domain: 'general' or 'medical'
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.domain = domain
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.vocab = checkpoint['vocab']
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        # Build model
        if domain == 'general':
            region_encoder = RegionFeatureExtractor(
                pretrained=True,
                num_regions=36,
                feature_dim=2048,
                device=device
            ).to(device)
        else:
            region_encoder = MedicalRegionExtractor(
                pretrained=True,
                num_regions=36,
                feature_dim=2048,
                device=device
            ).to(device)
        
        decoder = RegionAwareTransformerDecoder(
            vocab_size=len(self.vocab),
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=50 if domain == 'general' else 200,
            dropout=0.1,
            pad_token_id=self.vocab['<PAD>']
        ).to(device)
        
        self.model = CaptioningModel(
            region_encoder=region_encoder,
            decoder=decoder,
            region_feature_dim=2048,
            decoder_dim=512
        ).to(device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image preprocessing
        if domain == 'general':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
    
    def generate_caption(self, image_path, beam_size=3, return_attention=False):
        """
        Generate caption for an image.
        
        Args:
            image_path: Path to image file
            beam_size: Beam size for generation
            return_attention: Whether to return attention weights
            
        Returns:
            caption: Generated caption string
            attention_weights: (Optional) Attention weights
        """
        # Load and preprocess image
        image = Image.open(image_path)
        if self.domain == 'general':
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids, attention, _ = self.model.generate(
                image_tensor,
                start_token_id=self.vocab['<START>'],
                end_token_id=self.vocab['<END>'],
                max_len=50 if self.domain == 'general' else 200,
                beam_size=beam_size,
                return_attention=return_attention
            )
        
        # Convert to text
        caption_words = []
        for token_id in generated_ids[0].cpu().numpy():
            if token_id == self.vocab['<END>']:
                break
            if token_id in self.idx_to_word and token_id not in [
                self.vocab['<PAD>'], self.vocab['<START>']
            ]:
                caption_words.append(self.idx_to_word[token_id])
        
        caption = ' '.join(caption_words)
        
        if return_attention:
            return caption, attention
        return caption
    
    def visualize(self, image_path, beam_size=3):
        """
        Generate caption and visualize.
        
        Args:
            image_path: Path to image
            beam_size: Beam size for generation
        """
        # Generate caption
        caption = self.generate_caption(image_path, beam_size=beam_size)
        
        # Load original image for display
        image = Image.open(image_path)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Generated Caption:\n{caption}", fontsize=12, wrap=True)
        plt.tight_layout()
        plt.show()
        
        return caption


# Example usage:
if __name__ == '__main__':
    # General Domain Example
    print("=" * 80)
    print("GENERAL DOMAIN EXAMPLE")
    print("=" * 80)
    
    general_captioner = ImageCaptioner(
        checkpoint_path='../checkpoints/general/best_rl_model.pth',
        domain='general',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example image
    example_image = '../data/COCO/val2014/COCO_val2014_000000000042.jpg'
    
    if os.path.exists(example_image):
        caption = general_captioner.visualize(example_image, beam_size=3)
        print(f"\nGenerated: {caption}")
    else:
        print(f"Example image not found: {example_image}")
    
    # Medical Domain Example
    print("\n" + "=" * 80)
    print("MEDICAL DOMAIN EXAMPLE")
    print("=" * 80)
    
    medical_captioner = ImageCaptioner(
        checkpoint_path='../checkpoints/medical/best_rl_model.pth',
        domain='medical',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example medical image
    example_medical = '../data/MIMIC-CXR/example.dcm'
    
    if os.path.exists(example_medical):
        report = medical_captioner.visualize(example_medical, beam_size=3)
        print(f"\nGenerated Report: {report}")
    else:
        print(f"Example medical image not found: {example_medical}")
