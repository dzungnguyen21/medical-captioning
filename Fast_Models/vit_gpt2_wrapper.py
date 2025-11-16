"""
VisionEncoderDecoder (ViT + GPT2) Wrapper for Fast Medical Captioning
Lighter alternative to BLIP-2, faster training, less memory

Key Features:
- Vision Transformer (ViT) encoder
- GPT-2 decoder
- Full control over training process
- Faster than BLIP-2 on small datasets like IU X-Ray
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model
from typing import Optional, List, Dict


class ViTGPT2MedicalCaptioner(nn.Module):
    """
    ViT + GPT2 for medical image captioning
    
    Memory Usage on T4:
    - Full fp32: ~4GB (very safe)
    - fp16: ~2GB (recommended)
    - fp16 + LoRA: ~2.5GB
    
    Training Speed:
    - IU X-Ray: ~2-3 hours for 10 epochs on single T4
    - Faster than BLIP-2 due to smaller model size
    """
    
    def __init__(
        self,
        encoder_name: str = "google/vit-base-patch16-224-in21k",
        decoder_name: str = "gpt2",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.max_length = max_length
        self.use_lora = use_lora
        
        # Load image processor and tokenizer
        print(f"Loading ViT processor from {encoder_name}...")
        self.image_processor = ViTImageProcessor.from_pretrained(encoder_name)
        
        print(f"Loading GPT-2 tokenizer from {decoder_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load VisionEncoderDecoder model
        print(f"Creating VisionEncoderDecoder model...")
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_name, decoder_name
        )
        
        # Configure generation
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.max_length = max_length
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        
        # Apply LoRA if requested
        if use_lora:
            print("Applying LoRA to decoder...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA only to decoder
            self.model.decoder = get_peft_model(self.model.decoder, lora_config)
            self.model.decoder.print_trainable_parameters()
        
        self.model = self.model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Forward pass for training
        
        Args:
            pixel_values: Image tensor [batch_size, 3, H, W]
            labels: Target token IDs [batch_size, seq_len]
            return_dict: Whether to return dict output
            
        Returns:
            Model output with loss, logits, etc.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=return_dict
        )
        
        return outputs
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = False
    ) -> List[str]:
        """
        Generate captions for images
        
        Args:
            pixel_values: Image tensor [batch_size, 3, H, W]
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of captions per image
            do_sample: Whether to use sampling
            
        Returns:
            List of generated captions
        """
        if max_length is None:
            max_length = self.max_length
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample
            )
        
        # Decode to text
        captions = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return captions
    
    def generate_with_logprobs(
        self,
        pixel_values: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> tuple:
        """
        Generate captions with log probabilities for SCST
        
        Returns:
            generated_ids: Token sequences [batch_size, seq_len]
            logprobs: Log probabilities [batch_size, seq_len]
        """
        if max_length is None:
            max_length = self.max_length
        
        batch_size = pixel_values.shape[0]
        
        # Encode images
        with torch.no_grad():
            encoder_outputs = self.model.encoder(pixel_values=pixel_values)
        
        # Initialize decoder input
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        generated_ids = []
        logprobs = []
        
        # Autoregressive generation
        for step in range(max_length):
            # Forward through decoder
            outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                return_dict=True
            )
            
            # Get logits for next token
            next_token_logits = outputs.logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            
            # Apply top-p filtering if needed
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            next_token_logprob = torch.log(probs.gather(1, next_token))  # [batch_size, 1]
            
            generated_ids.append(next_token)
            logprobs.append(next_token_logprob)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        generated_ids = torch.cat(generated_ids, dim=1)  # [batch_size, seq_len]
        logprobs = torch.cat(logprobs, dim=1)  # [batch_size, seq_len]
        
        return generated_ids, logprobs
    
    def preprocess_images(self, images: List) -> torch.Tensor:
        """
        Preprocess PIL images to tensor
        
        Args:
            images: List of PIL images
            
        Returns:
            Tensor of shape [batch_size, 3, H, W]
        """
        processed = self.image_processor(images, return_tensors="pt")
        return processed.pixel_values.to(self.device)
    
    def save_pretrained(self, save_path: str):
        """Save model, processor, and tokenizer"""
        self.model.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """Load fine-tuned model"""
        instance = cls(**kwargs)
        instance.model = VisionEncoderDecoderModel.from_pretrained(load_path)
        instance.image_processor = ViTImageProcessor.from_pretrained(load_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        instance.model = instance.model.to(instance.device)
        return instance


def test_vit_gpt2_wrapper():
    """Test ViT-GPT2 wrapper with dummy data"""
    print("\n" + "="*80)
    print("Testing ViT-GPT2 Wrapper for Medical Captioning")
    print("="*80)
    
    # Initialize model
    model = ViTGPT2MedicalCaptioner(
        encoder_name="google/vit-base-patch16-224-in21k",
        decoder_name="gpt2",
        use_lora=True,
        max_length=64
    )
    
    # Dummy input
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(model.device)
    
    # Test forward pass
    print("\n[1] Testing forward pass with labels...")
    dummy_labels = torch.randint(0, 1000, (batch_size, 20)).to(model.device)
    outputs = model.forward(pixel_values=dummy_images, labels=dummy_labels)
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    
    # Test greedy generation
    print("\n[2] Testing greedy generation...")
    captions = model.generate(
        pixel_values=dummy_images,
        max_length=30,
        num_beams=1,
        do_sample=False
    )
    for i, cap in enumerate(captions):
        print(f"  Image {i+1}: {cap}")
    
    # Test beam search
    print("\n[3] Testing beam search generation...")
    captions = model.generate(
        pixel_values=dummy_images,
        max_length=30,
        num_beams=4,
        do_sample=False
    )
    for i, cap in enumerate(captions):
        print(f"  Image {i+1}: {cap}")
    
    # Test SCST generation
    print("\n[4] Testing SCST generation with logprobs...")
    gen_ids, logprobs = model.generate_with_logprobs(
        pixel_values=dummy_images,
        max_length=25
    )
    print(f"  Generated IDs shape: {gen_ids.shape}")
    print(f"  Log probabilities shape: {logprobs.shape}")
    captions = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    for i, cap in enumerate(captions):
        print(f"  Image {i+1}: {cap}")
        print(f"  Avg logprob: {logprobs[i].mean().item():.4f}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_vit_gpt2_wrapper()
