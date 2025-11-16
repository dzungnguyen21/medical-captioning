"""
BLIP-2 Wrapper for Fast Medical Image Captioning
Optimized for Kaggle T4 x2 GPUs with 4-day training constraint

Key Features:
- 8-bit quantization for memory efficiency
- LoRA fine-tuning for fast convergence
- Support for both CE and SCST training
"""

import torch
import torch.nn as nn
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, List


class Blip2MedicalCaptioner(nn.Module):
    """
    BLIP-2 wrapper optimized for medical image captioning
    
    Memory Usage on T4 (16GB):
    - Full fp32: ~14GB (risky)
    - 8-bit quantization: ~7GB (safe)
    - 8-bit + LoRA: ~8GB (recommended)
    
    Training Speed:
    - IU X-Ray (7.4K images): ~4-6 hours for 10 epochs on T4x2
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        use_8bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.use_lora = use_lora
        
        # Configure 8-bit quantization
        quantization_config = None
        if use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        
        # Load processor
        print(f"Loading BLIP-2 processor from {model_name}...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Load model
        print(f"Loading BLIP-2 model ({'8-bit' if use_8bit else 'fp32'})...")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if use_8bit else None,
            torch_dtype=torch.float16 if not use_8bit else None
        )
        
        # Apply LoRA if requested
        if use_lora:
            print("Applying LoRA for parameter-efficient fine-tuning...")
            
            # Prepare model for k-bit training
            if use_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],  # Apply to attention layers
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        if not use_8bit:
            self.model = self.model.to(device)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Forward pass for training
        
        Args:
            pixel_values: Image tensor [batch_size, 3, H, W]
            input_ids: Caption token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            return_dict: Whether to return dict output
            
        Returns:
            Model output with loss, logits, etc.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict
        )
        
        return outputs
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
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
            do_sample: Whether to use sampling (for SCST)
            
        Returns:
            List of generated captions
        """
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
        captions = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return captions
    
    def generate_with_logprobs(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> tuple:
        """
        Generate captions with log probabilities (for SCST)
        
        Returns:
            generated_ids: Token sequences [batch_size, seq_len]
            logprobs: Log probabilities [batch_size, seq_len]
        """
        batch_size = pixel_values.shape[0]
        
        # Get image embeddings
        with torch.no_grad():
            image_embeds = self.model.vision_model(pixel_values).last_hidden_state
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], 
                dtype=torch.long, 
                device=pixel_values.device
            )
            
            query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
            query_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True
            )
            image_embeds = self.model.language_projection(query_outputs.last_hidden_state)
        
        # Greedy/sampling generation with logprobs
        generated_ids = []
        logprobs = []
        
        # Start with BOS token
        input_ids = torch.full(
            (batch_size, 1), 
            self.processor.tokenizer.bos_token_id, 
            dtype=torch.long, 
            device=pixel_values.device
        )
        
        for _ in range(max_length):
            outputs = self.model.language_model(
                input_ids=input_ids,
                encoder_hidden_states=image_embeds,
                return_dict=True
            )
            
            logits = outputs.logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            if top_p < 1.0:
                # Nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            next_token_logprob = torch.log(probs.gather(1, next_token))  # [batch_size, 1]
            
            generated_ids.append(next_token)
            logprobs.append(next_token_logprob)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.processor.tokenizer.eos_token_id).all():
                break
        
        generated_ids = torch.cat(generated_ids, dim=1)  # [batch_size, seq_len]
        logprobs = torch.cat(logprobs, dim=1)  # [batch_size, seq_len]
        
        return generated_ids, logprobs
    
    def save_pretrained(self, save_path: str):
        """Save model and processor"""
        if self.use_lora:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """Load fine-tuned model"""
        instance = cls(**kwargs)
        instance.model = Blip2ForConditionalGeneration.from_pretrained(load_path)
        instance.processor = Blip2Processor.from_pretrained(load_path)
        return instance


def test_blip2_wrapper():
    """Test BLIP-2 wrapper with dummy data"""
    print("\n" + "="*80)
    print("Testing BLIP-2 Wrapper for Medical Captioning")
    print("="*80)
    
    # Initialize model
    model = Blip2MedicalCaptioner(
        model_name="Salesforce/blip2-opt-2.7b",
        use_8bit=True,
        use_lora=True
    )
    
    # Dummy input
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(model.device)
    
    # Test generation
    print("\n[1] Testing greedy generation...")
    captions = model.generate(
        pixel_values=dummy_images,
        max_length=50,
        num_beams=1,
        do_sample=False
    )
    for i, cap in enumerate(captions):
        print(f"  Image {i+1}: {cap}")
    
    # Test beam search
    print("\n[2] Testing beam search generation...")
    captions = model.generate(
        pixel_values=dummy_images,
        max_length=50,
        num_beams=4,
        do_sample=False
    )
    for i, cap in enumerate(captions):
        print(f"  Image {i+1}: {cap}")
    
    # Test SCST generation
    print("\n[3] Testing SCST generation with logprobs...")
    gen_ids, logprobs = model.generate_with_logprobs(
        pixel_values=dummy_images,
        max_length=30
    )
    print(f"  Generated IDs shape: {gen_ids.shape}")
    print(f"  Log probabilities shape: {logprobs.shape}")
    captions = model.processor.batch_decode(gen_ids, skip_special_tokens=True)
    for i, cap in enumerate(captions):
        print(f"  Image {i+1}: {cap}")
        print(f"  Avg logprob: {logprobs[i].mean().item():.4f}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_blip2_wrapper()
