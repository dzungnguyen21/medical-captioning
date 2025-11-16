"""
Keyword-Based Reward Function for Medical Image Captioning
Fast alternative to RadGraph for hallucination detection

Strategy:
- CIDEr score for language quality
- Keyword matching for factual accuracy
- Penalty for hallucinated medical terms
"""

import re
from typing import List, Dict, Set, Tuple
import torch
import numpy as np
from pycocoevalcap.cider.cider import Cider


# Medical keywords from RadLex and common chest X-ray findings
MEDICAL_KEYWORDS = {
    # Anatomical structures
    "anatomy": {
        "lung", "lungs", "heart", "cardiac", "mediastinum", "diaphragm",
        "trachea", "hilum", "hila", "pleura", "pleural", "costophrenic",
        "aorta", "aortic", "vascular", "vasculature", "clavicle", "rib", "ribs",
        "chest", "thorax", "thoracic", "pulmonary", "cardiopulmonary"
    },
    
    # Pathological findings
    "pathology": {
        "pneumonia", "consolidation", "infiltrate", "opacity", "opacities",
        "edema", "congestion", "effusion", "pneumothorax", "atelectasis",
        "nodule", "nodules", "mass", "lesion", "cavity", "cavitation",
        "cardiomegaly", "enlarged", "enlargement", "hyperinflation",
        "emphysema", "fibrosis", "scarring", "granuloma", "calcification"
    },
    
    # Descriptors
    "descriptors": {
        "normal", "clear", "unremarkable", "stable", "improved", "worsened",
        "bilateral", "unilateral", "right", "left", "upper", "lower", "middle",
        "lobe", "lobar", "focal", "diffuse", "patchy", "scattered",
        "mild", "moderate", "severe", "marked", "prominent", "increased", "decreased"
    },
    
    # Devices/Hardware
    "devices": {
        "tube", "line", "catheter", "port", "pacemaker", "defibrillator",
        "stent", "valve", "prosthetic", "hardware", "device"
    }
}

# Flatten all keywords
ALL_MEDICAL_KEYWORDS = set()
for category in MEDICAL_KEYWORDS.values():
    ALL_MEDICAL_KEYWORDS.update(category)


class KeywordRewardFunction:
    """
    Fast keyword-based reward for medical captioning
    
    Reward = α * CIDEr + β * Keyword_F1 - γ * Hallucination_Penalty
    
    Where:
    - CIDEr: Language quality score
    - Keyword_F1: F1 between predicted and ground truth medical keywords
    - Hallucination_Penalty: Penalty for keywords not in GT
    """
    
    def __init__(
        self,
        cider_weight: float = 1.0,
        keyword_weight: float = 0.5,
        hallucination_penalty: float = 0.3,
        custom_keywords: Optional[Set[str]] = None
    ):
        self.cider_weight = cider_weight
        self.keyword_weight = keyword_weight
        self.hallucination_penalty = hallucination_penalty
        
        # Medical keywords
        self.medical_keywords = ALL_MEDICAL_KEYWORDS.copy()
        if custom_keywords:
            self.medical_keywords.update(custom_keywords)
        
        # CIDEr scorer
        self.cider_scorer = Cider()
    
    def extract_keywords(self, text: str) -> Set[str]:
        """
        Extract medical keywords from text
        
        Args:
            text: Caption text
            
        Returns:
            Set of medical keywords found
        """
        # Lowercase and tokenize
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Find medical keywords
        keywords = set()
        for word in words:
            if word in self.medical_keywords:
                keywords.add(word)
        
        # Also check for multi-word terms (e.g., "pleural effusion")
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self.medical_keywords:
                keywords.add(bigram)
        
        return keywords
    
    def compute_keyword_f1(
        self,
        pred_keywords: Set[str],
        gt_keywords: Set[str]
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for keywords
        
        Args:
            pred_keywords: Predicted keywords
            gt_keywords: Ground truth keywords
            
        Returns:
            Dict with precision, recall, f1
        """
        if len(pred_keywords) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        if len(gt_keywords) == 0:
            # No GT keywords, perfect if pred also empty
            return {
                "precision": 1.0 if len(pred_keywords) == 0 else 0.0,
                "recall": 1.0,
                "f1": 1.0 if len(pred_keywords) == 0 else 0.0
            }
        
        # Compute metrics
        true_positives = len(pred_keywords & gt_keywords)
        precision = true_positives / len(pred_keywords) if len(pred_keywords) > 0 else 0.0
        recall = true_positives / len(gt_keywords) if len(gt_keywords) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def compute_hallucination_penalty(
        self,
        pred_keywords: Set[str],
        gt_keywords: Set[str]
    ) -> float:
        """
        Compute penalty for hallucinated medical keywords
        
        Hallucination = keywords in prediction but not in GT
        
        Args:
            pred_keywords: Predicted keywords
            gt_keywords: Ground truth keywords
            
        Returns:
            Hallucination rate (0 to 1)
        """
        if len(pred_keywords) == 0:
            return 0.0
        
        hallucinated = pred_keywords - gt_keywords
        return len(hallucinated) / len(pred_keywords)
    
    def compute_reward(
        self,
        predictions: List[str],
        references: List[List[str]],
        return_components: bool = False
    ) -> np.ndarray:
        """
        Compute reward for batch of predictions
        
        Args:
            predictions: List of predicted captions
            references: List of reference captions (each can have multiple refs)
            return_components: Whether to return reward components
            
        Returns:
            rewards: Array of shape [batch_size]
            (optional) components: Dict of reward components
        """
        batch_size = len(predictions)
        
        # Compute CIDEr scores
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        
        cider_scores, _ = self.cider_scorer.compute_score(gts, res)
        if isinstance(cider_scores, float):
            cider_scores = [cider_scores]
        cider_scores = np.array(cider_scores)
        
        # Compute keyword-based scores
        keyword_f1_scores = []
        hallucination_scores = []
        
        for pred, refs in zip(predictions, references):
            pred_keywords = self.extract_keywords(pred)
            
            # Combine keywords from all references
            gt_keywords = set()
            for ref in refs:
                gt_keywords.update(self.extract_keywords(ref))
            
            # Keyword F1
            kw_metrics = self.compute_keyword_f1(pred_keywords, gt_keywords)
            keyword_f1_scores.append(kw_metrics["f1"])
            
            # Hallucination penalty
            hall_penalty = self.compute_hallucination_penalty(pred_keywords, gt_keywords)
            hallucination_scores.append(hall_penalty)
        
        keyword_f1_scores = np.array(keyword_f1_scores)
        hallucination_scores = np.array(hallucination_scores)
        
        # Combine rewards
        rewards = (
            self.cider_weight * cider_scores +
            self.keyword_weight * keyword_f1_scores -
            self.hallucination_penalty * hallucination_scores
        )
        
        if return_components:
            components = {
                "cider": cider_scores,
                "keyword_f1": keyword_f1_scores,
                "hallucination": hallucination_scores,
                "total_reward": rewards
            }
            return rewards, components
        
        return rewards
    
    def __call__(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> torch.Tensor:
        """
        Compute rewards as PyTorch tensor
        
        Args:
            predictions: List of predicted captions
            references: List of reference captions
            
        Returns:
            Tensor of rewards [batch_size]
        """
        rewards = self.compute_reward(predictions, references)
        return torch.from_numpy(rewards).float()


def test_keyword_reward():
    """Test keyword-based reward function"""
    print("\n" + "="*80)
    print("Testing Keyword-Based Reward Function")
    print("="*80)
    
    # Initialize reward function
    reward_fn = KeywordRewardFunction(
        cider_weight=1.0,
        keyword_weight=0.5,
        hallucination_penalty=0.3
    )
    
    # Test cases
    test_cases = [
        {
            "name": "Perfect match",
            "prediction": "The lungs are clear. No pneumonia or effusion.",
            "reference": ["The lungs are clear. No pneumonia or pleural effusion."]
        },
        {
            "name": "Hallucination (fake finding)",
            "prediction": "The lungs show pneumonia and mass. Cardiomegaly present.",
            "reference": ["The lungs are clear. Heart is normal size."]
        },
        {
            "name": "Missing findings",
            "prediction": "The lungs are clear.",
            "reference": ["Bilateral pneumonia with pleural effusion."]
        },
        {
            "name": "Normal variation",
            "prediction": "No acute cardiopulmonary abnormality.",
            "reference": ["Lungs are clear. Heart size is normal."]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}] {case['name']}")
        print(f"  Prediction: {case['prediction']}")
        print(f"  Reference:  {case['reference'][0]}")
        
        # Extract keywords
        pred_kw = reward_fn.extract_keywords(case['prediction'])
        ref_kw = reward_fn.extract_keywords(case['reference'][0])
        print(f"  Pred keywords: {pred_kw}")
        print(f"  Ref keywords:  {ref_kw}")
        
        # Compute reward
        reward, components = reward_fn.compute_reward(
            [case['prediction']],
            [case['reference']],
            return_components=True
        )
        
        print(f"  CIDEr: {components['cider'][0]:.3f}")
        print(f"  Keyword F1: {components['keyword_f1'][0]:.3f}")
        print(f"  Hallucination: {components['hallucination'][0]:.3f}")
        print(f"  Total Reward: {components['total_reward'][0]:.3f}")
    
    # Test batch processing
    print("\n" + "-"*80)
    print("Testing batch processing...")
    
    predictions = [tc["prediction"] for tc in test_cases]
    references = [tc["reference"] for tc in test_cases]
    
    rewards = reward_fn(predictions, references)
    print(f"Batch rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards.numpy()}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    from typing import Optional
    test_keyword_reward()
