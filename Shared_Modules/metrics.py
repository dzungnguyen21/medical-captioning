"""
Evaluation Metrics Module
Implements all metrics for comprehensive evaluation
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from collections import defaultdict
import json


class CaptionMetrics:
    """
    Comprehensive metrics for image captioning evaluation.
    Includes: BLEU, METEOR, ROUGE-L, CIDEr, SPICE
    """
    
    def __init__(self):
        self.scorers = {
            'BLEU': Bleu(4),
            'METEOR': Meteor(),
            'ROUGE_L': Rouge(),
            'CIDEr': Cider(),
        }
        
        # SPICE requires Java, so make it optional
        try:
            self.scorers['SPICE'] = Spice()
        except:
            print("Warning: SPICE not available (requires Java)")
    
    def compute_scores(
        self,
        generated_captions: Dict[int, List[str]],
        reference_captions: Dict[int, List[str]]
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            generated_captions: Dict mapping image_id -> [generated_caption]
            reference_captions: Dict mapping image_id -> [ref1, ref2, ...]
            
        Returns:
            Dictionary of metric scores
        """
        scores = {}
        
        for metric_name, scorer in self.scorers.items():
            score, _ = scorer.compute_score(reference_captions, generated_captions)
            
            if metric_name == 'BLEU':
                # BLEU returns 4 scores
                for i, s in enumerate(score):
                    scores[f'BLEU-{i+1}'] = s
            else:
                scores[metric_name] = score
        
        return scores


class POPEMetrics:
    """
    POPE (Polling-based Object Probing Evaluation)
    Evaluates whether the model can correctly identify objects present/absent in images.
    """
    
    def __init__(self, object_classes: List[str]):
        """
        Args:
            object_classes: List of object class names
        """
        self.object_classes = object_classes
    
    def create_pope_questions(
        self,
        image_id: int,
        detected_objects: List[str],
        num_questions: int = 10
    ) -> List[Dict]:
        """
        Create POPE questions for an image.
        
        Returns:
            List of questions: [{"question": "Is there a dog?", "answer": "yes"}]
        """
        questions = []
        
        # Positive samples (objects that are present)
        positive_objects = detected_objects[:num_questions // 2]
        for obj in positive_objects:
            questions.append({
                'image_id': image_id,
                'question': f'Is there a {obj}?',
                'answer': 'yes',
                'object': obj
            })
        
        # Negative samples (objects that are not present)
        absent_objects = [obj for obj in self.object_classes if obj not in detected_objects]
        negative_objects = np.random.choice(
            absent_objects,
            size=min(num_questions // 2, len(absent_objects)),
            replace=False
        )
        
        for obj in negative_objects:
            questions.append({
                'image_id': image_id,
                'question': f'Is there a {obj}?',
                'answer': 'no',
                'object': obj
            })
        
        return questions
    
    def evaluate_pope(
        self,
        generated_captions: Dict[int, str],
        pope_questions: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate POPE performance.
        
        Args:
            generated_captions: Dict mapping image_id -> generated_caption
            pope_questions: List of POPE questions
            
        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for q in pope_questions:
            image_id = q['image_id']
            object_name = q['object'].lower()
            true_answer = q['answer']
            
            # Check if object is mentioned in generated caption
            caption = generated_captions.get(image_id, '').lower()
            predicted_answer = 'yes' if object_name in caption else 'no'
            
            # Update confusion matrix
            if true_answer == 'yes' and predicted_answer == 'yes':
                true_positives += 1
            elif true_answer == 'yes' and predicted_answer == 'no':
                false_negatives += 1
            elif true_answer == 'no' and predicted_answer == 'yes':
                false_positives += 1
            elif true_answer == 'no' and predicted_answer == 'no':
                true_negatives += 1
        
        # Compute metrics
        accuracy = (true_positives + true_negatives) / len(pope_questions)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }


class PointingGameMetrics:
    """
    Pointing Game: Evaluates whether attention focuses on correct regions.
    Measures grounding accuracy.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: IoU threshold for considering a prediction correct
        """
        self.iou_threshold = iou_threshold
    
    def compute_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """
        Compute IoU between two boxes.
        
        Args:
            box1, box2: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def evaluate_pointing_game(
        self,
        attention_weights: torch.Tensor,
        region_boxes: torch.Tensor,
        ground_truth_boxes: torch.Tensor,
        caption_words: List[str],
        object_words: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate pointing game accuracy.
        
        Args:
            attention_weights: [seq_len, num_regions] attention weights
            region_boxes: [num_regions, 4] region bounding boxes
            ground_truth_boxes: [num_objects, 4] ground truth boxes for objects
            caption_words: List of words in caption
            object_words: List of object words to evaluate
            
        Returns:
            Dictionary with accuracy metrics
        """
        correct = 0
        total = 0
        
        for i, word in enumerate(caption_words):
            if word.lower() not in object_words:
                continue
            
            total += 1
            
            # Get attention for this word
            word_attention = attention_weights[i]  # [num_regions]
            
            # Get region with highest attention
            max_region_idx = word_attention.argmax().item()
            attended_box = region_boxes[max_region_idx].cpu().numpy()
            
            # Check if attended box overlaps with any ground truth box
            max_iou = 0
            for gt_box in ground_truth_boxes:
                iou = self.compute_iou(attended_box, gt_box.cpu().numpy())
                max_iou = max(max_iou, iou)
            
            if max_iou >= self.iou_threshold:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'pointing_accuracy': accuracy,
            'correct': correct,
            'total': total
        }


class MedicalMetrics:
    """
    Medical-specific metrics: RadGraph F1, CheXbert F1
    """
    
    def __init__(self, use_radgraph: bool = True, use_chexbert: bool = True):
        self.use_radgraph = use_radgraph
        self.use_chexbert = use_chexbert
        
        # Initialize RadGraph
        if use_radgraph:
            try:
                import radgraph
                self.radgraph = radgraph
            except ImportError:
                print("Warning: RadGraph not available")
                self.use_radgraph = False
        
        # Initialize CheXbert
        if use_chexbert:
            try:
                from chexbert import CheXbert
                self.chexbert = CheXbert()
            except ImportError:
                print("Warning: CheXbert not available")
                self.use_chexbert = False
    
    def compute_radgraph_f1(
        self,
        generated_captions: List[str],
        reference_captions: List[str]
    ) -> Dict[str, float]:
        """
        Compute RadGraph F1 scores.
        RadGraph extracts clinical entities and relations from radiology reports.
        """
        if not self.use_radgraph:
            return {'RadGraph_F1': 0.0}
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        count = 0
        
        for gen_cap, ref_cap in zip(generated_captions, reference_captions):
            try:
                # Parse captions to extract entities and relations
                gen_graph = self.radgraph.parse(gen_cap)
                ref_graph = self.radgraph.parse(ref_cap)
                
                # Extract entities
                gen_entities = set(gen_graph.get('entities', []))
                ref_entities = set(ref_graph.get('entities', []))
                
                # Compute precision, recall, F1
                if len(gen_entities) == 0 and len(ref_entities) == 0:
                    precision = recall = f1 = 1.0
                elif len(gen_entities) == 0 or len(ref_entities) == 0:
                    precision = recall = f1 = 0.0
                else:
                    true_positives = len(gen_entities & ref_entities)
                    precision = true_positives / len(gen_entities)
                    recall = true_positives / len(ref_entities)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                count += 1
                
            except Exception as e:
                print(f"RadGraph error: {e}")
                continue
        
        if count == 0:
            return {'RadGraph_F1': 0.0, 'RadGraph_Precision': 0.0, 'RadGraph_Recall': 0.0}
        
        return {
            'RadGraph_F1': total_f1 / count,
            'RadGraph_Precision': total_precision / count,
            'RadGraph_Recall': total_recall / count
        }
    
    def compute_chexbert_f1(
        self,
        generated_captions: List[str],
        reference_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute CheXbert F1 scores.
        CheXbert classifies 14 pathologies from radiology reports.
        
        Args:
            generated_captions: List of generated captions
            reference_labels: [num_samples, 14] binary labels for 14 pathologies
            
        Returns:
            Dictionary with F1 scores
        """
        if not self.use_chexbert:
            return {'CheXbert_F1': 0.0}
        
        try:
            # Extract labels from generated captions
            generated_labels = self.chexbert.label(generated_captions)
            
            # Compute F1 for each pathology
            f1_scores = []
            pathology_names = [
                'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
            ]
            
            results = {}
            
            for i, pathology in enumerate(pathology_names):
                gen_labels_i = generated_labels[:, i]
                ref_labels_i = reference_labels[:, i]
                
                # Compute F1
                true_positives = ((gen_labels_i == 1) & (ref_labels_i == 1)).sum()
                false_positives = ((gen_labels_i == 1) & (ref_labels_i == 0)).sum()
                false_negatives = ((gen_labels_i == 0) & (ref_labels_i == 1)).sum()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f1_scores.append(f1)
                results[f'CheXbert_F1_{pathology}'] = f1
            
            # Average F1
            results['CheXbert_F1'] = np.mean(f1_scores)
            
            return results
            
        except Exception as e:
            print(f"CheXbert error: {e}")
            return {'CheXbert_F1': 0.0}


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that combines all metrics.
    """
    
    def __init__(
        self,
        hallucination_detector=None,
        medical_detector=None,
        object_classes: List[str] = None,
        domain: str = 'general'
    ):
        """
        Args:
            hallucination_detector: HallucinationDetector instance
            medical_detector: MedicalHallucinationDetector instance
            object_classes: List of object class names
            domain: 'general' or 'medical'
        """
        self.domain = domain
        self.caption_metrics = CaptionMetrics()
        
        if domain == 'general':
            self.hallucination_detector = hallucination_detector
            if object_classes:
                self.pope_metrics = POPEMetrics(object_classes)
            self.pointing_game = PointingGameMetrics()
        elif domain == 'medical':
            self.medical_detector = medical_detector
            self.medical_metrics = MedicalMetrics()
    
    def evaluate(
        self,
        generated_captions: Dict[int, List[str]],
        reference_captions: Dict[int, List[str]],
        detected_objects: Dict[int, List[str]] = None,
        attention_weights: Dict[int, torch.Tensor] = None,
        reference_labels: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation.
        
        Returns:
            Dictionary with all metric scores
        """
        results = {}
        
        # 1. Standard NLG metrics (BLEU, METEOR, ROUGE, CIDEr)
        nlg_scores = self.caption_metrics.compute_scores(
            generated_captions, reference_captions
        )
        results.update(nlg_scores)
        
        # 2. Domain-specific metrics
        if self.domain == 'general':
            # CHAIR metrics
            if self.hallucination_detector and detected_objects:
                gen_caps_list = [generated_captions[i][0] for i in sorted(generated_captions.keys())]
                det_objs_list = [set(detected_objects[i]) for i in sorted(detected_objects.keys())]
                
                chair_scores = self.hallucination_detector.compute_chair_metrics(
                    gen_caps_list, det_objs_list
                )
                results.update(chair_scores)
            
            # POPE metrics (if applicable)
            # This would require pre-generated POPE questions
            
            # Pointing Game (if attention weights provided)
            if attention_weights:
                # Implement pointing game evaluation
                pass
        
        elif self.domain == 'medical':
            # RadGraph F1
            if self.medical_metrics.use_radgraph:
                gen_caps_list = [generated_captions[i][0] for i in sorted(generated_captions.keys())]
                ref_caps_list = [reference_captions[i][0] for i in sorted(reference_captions.keys())]
                
                radgraph_scores = self.medical_metrics.compute_radgraph_f1(
                    gen_caps_list, ref_caps_list
                )
                results.update(radgraph_scores)
            
            # CheXbert F1
            if self.medical_metrics.use_chexbert and reference_labels is not None:
                gen_caps_list = [generated_captions[i][0] for i in sorted(generated_captions.keys())]
                
                chexbert_scores = self.medical_metrics.compute_chexbert_f1(
                    gen_caps_list, reference_labels
                )
                results.update(chexbert_scores)
        
        return results
