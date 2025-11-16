"""
Reward Functions for Reinforcement Learning
Includes CIDEr, CHAIR penalty (General) and RadGraph F1 (Medical)
"""

import torch
from typing import List, Dict
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import numpy as np


class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, idx_to_word: Dict[int, str]):
        self.idx_to_word = idx_to_word
    
    def tokens_to_caption(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs to caption string."""
        words = []
        for token_id in token_ids:
            token_id = token_id.item()
            if token_id in self.idx_to_word and token_id != 0:  # Skip padding
                words.append(self.idx_to_word[token_id])
        return ' '.join(words)
    
    def __call__(
        self,
        generated_ids: torch.Tensor,
        references: List[str],
        encoder_output: Dict = None
    ) -> torch.Tensor:
        """
        Compute rewards.
        
        Args:
            generated_ids: [batch_size, seq_len]
            references: List of reference caption strings
            encoder_output: Output from region encoder (for object-based rewards)
            
        Returns:
            rewards: [batch_size]
        """
        raise NotImplementedError


class CIDErReward(RewardFunction):
    """
    CIDEr reward function.
    CIDEr (Consensus-based Image Description Evaluation) is the standard metric
    for image captioning that measures consensus with human descriptions.
    """
    
    def __init__(self, idx_to_word: Dict[int, str]):
        super().__init__(idx_to_word)
        self.cider_scorer = Cider()
    
    def __call__(
        self,
        generated_ids: torch.Tensor,
        references: List[str],
        encoder_output: Dict = None
    ) -> torch.Tensor:
        batch_size = generated_ids.size(0)
        device = generated_ids.device
        
        # Convert to captions
        gts = {}
        res = {}
        
        for i in range(batch_size):
            caption = self.tokens_to_caption(generated_ids[i])
            res[i] = [caption]
            gts[i] = [references[i]] if isinstance(references[i], str) else references[i]
        
        # Compute CIDEr scores
        _, scores = self.cider_scorer.compute_score(gts, res)
        
        # Convert to tensor
        rewards = torch.tensor(scores, dtype=torch.float32, device=device)
        
        return rewards


class GeneralReward(RewardFunction):
    """
    Reward function for General domain:
    R = α * CIDEr + β * (1 - CHAIR_score)
    
    Encourages fluent captions while penalizing hallucinated objects.
    """
    
    def __init__(
        self,
        idx_to_word: Dict[int, str],
        hallucination_detector,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        """
        Args:
            idx_to_word: Vocabulary mapping
            hallucination_detector: HallucinationDetector instance
            alpha: Weight for CIDEr reward
            beta: Weight for hallucination penalty
        """
        super().__init__(idx_to_word)
        self.hallucination_detector = hallucination_detector
        self.alpha = alpha
        self.beta = beta
        self.cider_scorer = Cider()
    
    def __call__(
        self,
        generated_ids: torch.Tensor,
        references: List[str],
        encoder_output: Dict = None
    ) -> torch.Tensor:
        batch_size = generated_ids.size(0)
        device = generated_ids.device
        
        # 1. Compute CIDEr reward
        gts = {}
        res = {}
        captions = []
        
        for i in range(batch_size):
            caption = self.tokens_to_caption(generated_ids[i])
            captions.append(caption)
            res[i] = [caption]
            gts[i] = [references[i]] if isinstance(references[i], str) else references[i]
        
        _, cider_scores = self.cider_scorer.compute_score(gts, res)
        cider_rewards = torch.tensor(cider_scores, dtype=torch.float32, device=device)
        
        # 2. Compute object fidelity reward (anti-hallucination)
        if encoder_output is not None and 'labels' in encoder_output:
            object_rewards = self.hallucination_detector.compute_object_fidelity_reward(
                generated_ids, encoder_output['labels']
            )
        else:
            # Fallback: compute per-caption CHAIR and convert to reward
            object_rewards = torch.zeros(batch_size, device=device)
            
            if encoder_output is not None and 'labels' in encoder_output:
                detected_objects_list = []
                for labels in encoder_output['labels']:
                    detected_set = set()
                    for label_id in labels:
                        label_id = label_id.item()
                        # Map label_id to object name (assuming COCO classes)
                        detected_set.add(str(label_id))  # Simplified
                    detected_objects_list.append(detected_set)
                
                # Compute CHAIR for this batch
                chair_metrics = self.hallucination_detector.compute_chair_metrics(
                    captions, detected_objects_list
                )
                
                # Convert CHAIR to reward (lower CHAIR = higher reward)
                chair_score = chair_metrics['CHAIR_i']
                object_reward = 1.0 - chair_score
                object_rewards = torch.full(
                    (batch_size,), object_reward, dtype=torch.float32, device=device
                )
        
        # 3. Combine rewards
        total_rewards = self.alpha * cider_rewards + self.beta * object_rewards
        
        return total_rewards


class MedicalReward(RewardFunction):
    """
    Reward function for Medical domain:
    R = α * CIDEr + β * RadGraph_F1
    
    Encourages clinically accurate captions with proper entity grounding.
    """
    
    def __init__(
        self,
        idx_to_word: Dict[int, str],
        medical_detector,
        alpha: float = 1.0,
        beta: float = 2.0
    ):
        """
        Args:
            idx_to_word: Vocabulary mapping
            medical_detector: MedicalHallucinationDetector instance
            alpha: Weight for CIDEr reward
            beta: Weight for RadGraph F1 reward
        """
        super().__init__(idx_to_word)
        self.medical_detector = medical_detector
        self.alpha = alpha
        self.beta = beta
        self.cider_scorer = Cider()
    
    def __call__(
        self,
        generated_ids: torch.Tensor,
        references: List[str],
        encoder_output: Dict = None
    ) -> torch.Tensor:
        batch_size = generated_ids.size(0)
        device = generated_ids.device
        
        # 1. Compute CIDEr reward
        gts = {}
        res = {}
        captions = []
        
        for i in range(batch_size):
            caption = self.tokens_to_caption(generated_ids[i])
            captions.append(caption)
            res[i] = [caption]
            gts[i] = [references[i]] if isinstance(references[i], str) else references[i]
        
        _, cider_scores = self.cider_scorer.compute_score(gts, res)
        cider_rewards = torch.tensor(cider_scores, dtype=torch.float32, device=device)
        
        # 2. Compute RadGraph F1 reward
        radgraph_rewards = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            if self.medical_detector.use_radgraph:
                radgraph_f1 = self.medical_detector.compute_radgraph_reward(
                    captions[i], references[i]
                )
                radgraph_rewards[i] = radgraph_f1
        
        # 3. Combine rewards
        total_rewards = self.alpha * cider_rewards + self.beta * radgraph_rewards
        
        return total_rewards


class MultiReward(RewardFunction):
    """
    Combined reward with multiple metrics.
    Useful for ablation studies.
    """
    
    def __init__(
        self,
        idx_to_word: Dict[int, str],
        weights: Dict[str, float] = None
    ):
        """
        Args:
            idx_to_word: Vocabulary mapping
            weights: Dictionary of metric weights
                     e.g., {'cider': 1.0, 'bleu': 0.5, 'meteor': 0.5}
        """
        super().__init__(idx_to_word)
        
        if weights is None:
            weights = {'cider': 1.0}
        
        self.weights = weights
        
        # Initialize scorers
        self.scorers = {}
        if 'cider' in weights:
            self.scorers['cider'] = Cider()
        if 'bleu' in weights:
            self.scorers['bleu'] = Bleu(4)
        if 'meteor' in weights:
            self.scorers['meteor'] = Meteor()
        if 'rouge' in weights:
            self.scorers['rouge'] = Rouge()
    
    def __call__(
        self,
        generated_ids: torch.Tensor,
        references: List[str],
        encoder_output: Dict = None
    ) -> torch.Tensor:
        batch_size = generated_ids.size(0)
        device = generated_ids.device
        
        # Convert to captions
        gts = {}
        res = {}
        
        for i in range(batch_size):
            caption = self.tokens_to_caption(generated_ids[i])
            res[i] = [caption]
            gts[i] = [references[i]] if isinstance(references[i], str) else references[i]
        
        # Compute all metrics
        total_rewards = torch.zeros(batch_size, device=device)
        
        for metric_name, scorer in self.scorers.items():
            if metric_name in self.weights:
                weight = self.weights[metric_name]
                
                if metric_name == 'bleu':
                    # BLEU returns multiple scores
                    _, scores_list = scorer.compute_score(gts, res)
                    # Use BLEU-4
                    scores = scores_list[-1]
                else:
                    _, scores = scorer.compute_score(gts, res)
                
                metric_rewards = torch.tensor(scores, dtype=torch.float32, device=device)
                total_rewards += weight * metric_rewards
        
        return total_rewards
