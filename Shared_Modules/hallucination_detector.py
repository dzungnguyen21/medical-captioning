"""
Hallucination Detection Module
Compares generated words with detected object tags to prevent hallucination
"""

import torch
import torch.nn as nn
from typing import List, Set, Dict, Tuple
import nltk
from collections import defaultdict


class HallucinationDetector(nn.Module):
    """
    Module to detect and penalize hallucinated objects in generated captions.
    Works by comparing generated nouns/objects with detected visual objects.
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        object_class_names: List[str],
        use_synonym_matching: bool = True
    ):
        """
        Args:
            vocab: Vocabulary dictionary mapping words to indices
            object_class_names: List of object class names from detector (e.g., COCO classes)
            use_synonym_matching: Whether to use synonym matching for more robust detection
        """
        super().__init__()
        
        self.vocab = vocab
        self.idx_to_word = {v: k for k, v in vocab.items()}
        self.object_class_names = set(object_class_names)
        self.use_synonym_matching = use_synonym_matching
        
        # Download required NLTK data
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        if use_synonym_matching:
            from nltk.corpus import wordnet
            self.wordnet = wordnet
            
            # Build synonym map for object classes
            self.object_synonyms = self._build_synonym_map()
    
    def _build_synonym_map(self) -> Dict[str, Set[str]]:
        """
        Build a map of object classes to their synonyms using WordNet.
        """
        synonym_map = defaultdict(set)
        
        for obj_class in self.object_class_names:
            # Add the class itself
            synonym_map[obj_class].add(obj_class.lower())
            
            # Add synonyms from WordNet
            for syn in self.wordnet.synsets(obj_class):
                for lemma in syn.lemmas():
                    synonym_map[obj_class].add(lemma.name().lower().replace('_', ' '))
        
        return synonym_map
    
    def extract_nouns_and_objects(self, caption: str) -> Set[str]:
        """
        Extract nouns and potential object references from caption.
        
        Args:
            caption: Generated caption string
            
        Returns:
            Set of nouns/objects in the caption
        """
        # Tokenize and POS tag
        tokens = nltk.word_tokenize(caption.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract nouns (NN, NNS, NNP, NNPS)
        nouns = set()
        for word, pos in pos_tags:
            if pos.startswith('NN'):
                nouns.add(word)
        
        return nouns
    
    def compute_chair_metrics(
        self,
        generated_captions: List[str],
        detected_objects_list: List[Set[str]]
    ) -> Dict[str, float]:
        """
        Compute CHAIR (Caption Hallucination Assessment with Image Relevance) metrics.
        
        CHAIR_i: Proportion of hallucinated objects per caption (instance-level)
        CHAIR_s: Proportion of captions with at least one hallucination (sentence-level)
        
        Args:
            generated_captions: List of generated caption strings
            detected_objects_list: List of sets containing detected object class names
            
        Returns:
            Dictionary with CHAIR_i and CHAIR_s scores
        """
        total_objects = 0
        hallucinated_objects = 0
        captions_with_hallucination = 0
        
        for caption, detected_objects in zip(generated_captions, detected_objects_list):
            # Extract nouns from caption
            caption_objects = self.extract_nouns_and_objects(caption)
            
            if len(caption_objects) == 0:
                continue
            
            # Check each object against detected objects
            caption_hallucinations = 0
            for obj in caption_objects:
                is_hallucination = True
                
                # Direct match
                if obj in detected_objects:
                    is_hallucination = False
                
                # Synonym match (if enabled)
                elif self.use_synonym_matching:
                    for detected_obj in detected_objects:
                        if obj in self.object_synonyms.get(detected_obj, set()):
                            is_hallucination = False
                            break
                        # Also check reverse (if detected_obj is a synonym of obj)
                        if detected_obj in self.object_synonyms.get(obj, set()):
                            is_hallucination = False
                            break
                
                if is_hallucination:
                    hallucinated_objects += 1
                    caption_hallucinations += 1
            
            total_objects += len(caption_objects)
            
            if caption_hallucinations > 0:
                captions_with_hallucination += 1
        
        # Compute metrics
        chair_i = hallucinated_objects / total_objects if total_objects > 0 else 0.0
        chair_s = captions_with_hallucination / len(generated_captions) if len(generated_captions) > 0 else 0.0
        
        return {
            'CHAIR_i': chair_i,
            'CHAIR_s': chair_s,
            'num_hallucinated_objects': hallucinated_objects,
            'total_objects': total_objects,
            'num_captions_with_hallucination': captions_with_hallucination
        }
    
    def compute_object_fidelity_reward(
        self,
        generated_tokens: torch.Tensor,
        detected_labels: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute object fidelity reward for RL training.
        Rewards captions that mention detected objects, penalizes hallucinations.
        
        Args:
            generated_tokens: [batch_size, seq_len] generated token IDs
            detected_labels: List of detected object label tensors for each image
            
        Returns:
            rewards: [batch_size] reward values
        """
        batch_size = generated_tokens.size(0)
        rewards = torch.zeros(batch_size, device=generated_tokens.device)
        
        for i in range(batch_size):
            # Convert tokens to words
            caption_words = []
            for token_id in generated_tokens[i]:
                token_id = token_id.item()
                if token_id in self.idx_to_word:
                    caption_words.append(self.idx_to_word[token_id])
            
            caption = ' '.join(caption_words)
            
            # Extract objects from caption
            caption_objects = self.extract_nouns_and_objects(caption)
            
            if len(caption_objects) == 0:
                continue
            
            # Get detected object names for this image
            detected_object_names = set()
            if i < len(detected_labels):
                for label_id in detected_labels[i]:
                    label_id = label_id.item()
                    if 0 <= label_id < len(self.object_class_names):
                        detected_object_names.add(
                            list(self.object_class_names)[label_id].lower()
                        )
            
            # Compute reward
            correct_objects = 0
            hallucinated_objects = 0
            
            for obj in caption_objects:
                if obj in detected_object_names:
                    correct_objects += 1
                else:
                    # Check synonyms
                    is_hallucination = True
                    if self.use_synonym_matching:
                        for detected_obj in detected_object_names:
                            if obj in self.object_synonyms.get(detected_obj, set()):
                                correct_objects += 1
                                is_hallucination = False
                                break
                    
                    if is_hallucination:
                        hallucinated_objects += 1
            
            # Reward formula: encourage correct objects, heavily penalize hallucinations
            object_reward = correct_objects - 2.0 * hallucinated_objects
            object_reward = object_reward / len(caption_objects)  # Normalize
            
            rewards[i] = object_reward
        
        return rewards


class MedicalHallucinationDetector(nn.Module):
    """
    Medical-specific hallucination detector.
    Uses clinical entity extraction and RadGraph for grounding.
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        use_chexbert: bool = True,
        use_radgraph: bool = True
    ):
        """
        Args:
            vocab: Vocabulary dictionary
            use_chexbert: Whether to use CheXbert for clinical entity extraction
            use_radgraph: Whether to use RadGraph for entity grounding
        """
        super().__init__()
        
        self.vocab = vocab
        self.idx_to_word = {v: k for k, v in vocab.items()}
        self.use_chexbert = use_chexbert
        self.use_radgraph = use_radgraph
        
        # Medical entity categories
        self.anatomical_entities = {
            'lung', 'lungs', 'heart', 'cardiac', 'chest', 'thorax',
            'mediastinum', 'diaphragm', 'pleura', 'pleural', 'trachea',
            'bronchus', 'bronchi', 'aorta', 'hilum', 'hila', 'rib', 'ribs',
            'clavicle', 'spine', 'vertebra', 'costophrenic'
        }
        
        self.pathology_entities = {
            'pneumonia', 'effusion', 'edema', 'atelectasis', 'consolidation',
            'pneumothorax', 'cardiomegaly', 'nodule', 'mass', 'opacity',
            'infiltrate', 'congestion', 'lesion', 'abnormality', 'fracture',
            'emphysema', 'fibrosis', 'granuloma', 'calcification'
        }
        
        # Load CheXbert if available
        if use_chexbert:
            try:
                from chexbert import CheXbert
                self.chexbert = CheXbert()
            except ImportError:
                print("Warning: CheXbert not available. Install with: pip install chexbert")
                self.use_chexbert = False
        
        # Load RadGraph if available
        if use_radgraph:
            try:
                import radgraph
                self.radgraph = radgraph
            except ImportError:
                print("Warning: RadGraph not available. Install with: pip install radgraph")
                self.use_radgraph = False
    
    def extract_clinical_entities(self, caption: str) -> Dict[str, Set[str]]:
        """
        Extract clinical entities from caption.
        
        Returns:
            Dictionary with 'anatomical' and 'pathology' entities
        """
        tokens = set(caption.lower().split())
        
        entities = {
            'anatomical': tokens & self.anatomical_entities,
            'pathology': tokens & self.pathology_entities
        }
        
        return entities
    
    def compute_radgraph_reward(
        self,
        generated_caption: str,
        reference_caption: str
    ) -> float:
        """
        Compute RadGraph F1 score between generated and reference captions.
        RadGraph extracts clinical entities and relationships.
        
        Args:
            generated_caption: Generated caption
            reference_caption: Ground truth caption
            
        Returns:
            RadGraph F1 score
        """
        if not self.use_radgraph:
            return 0.0
        
        try:
            # Extract RadGraph annotations
            gen_graph = self.radgraph.parse(generated_caption)
            ref_graph = self.radgraph.parse(reference_caption)
            
            # Compute F1 score based on entity and relation matching
            # This is a simplified version; actual implementation would use RadGraph metrics
            gen_entities = set(gen_graph.get('entities', []))
            ref_entities = set(ref_graph.get('entities', []))
            
            if len(gen_entities) == 0 and len(ref_entities) == 0:
                return 1.0
            
            if len(gen_entities) == 0 or len(ref_entities) == 0:
                return 0.0
            
            # Compute precision and recall
            true_positives = len(gen_entities & ref_entities)
            precision = true_positives / len(gen_entities)
            recall = true_positives / len(ref_entities)
            
            # F1 score
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * precision * recall / (precision + recall)
            return f1
            
        except Exception as e:
            print(f"RadGraph computation failed: {e}")
            return 0.0
    
    def compute_medical_fidelity_reward(
        self,
        generated_tokens: torch.Tensor,
        reference_captions: List[str],
        detected_pathologies: List[Set[str]] = None
    ) -> torch.Tensor:
        """
        Compute medical fidelity reward for RL training.
        
        Args:
            generated_tokens: [batch_size, seq_len]
            reference_captions: List of ground truth captions
            detected_pathologies: List of detected pathology labels (if available)
            
        Returns:
            rewards: [batch_size]
        """
        batch_size = generated_tokens.size(0)
        rewards = torch.zeros(batch_size, device=generated_tokens.device)
        
        for i in range(batch_size):
            # Convert tokens to caption
            caption_words = []
            for token_id in generated_tokens[i]:
                token_id = token_id.item()
                if token_id in self.idx_to_word:
                    caption_words.append(self.idx_to_word[token_id])
            
            generated_caption = ' '.join(caption_words)
            reference_caption = reference_captions[i] if i < len(reference_captions) else ""
            
            # Compute RadGraph F1 if available
            if self.use_radgraph and reference_caption:
                radgraph_score = self.compute_radgraph_reward(
                    generated_caption, reference_caption
                )
                rewards[i] += radgraph_score
            
            # Compute entity-based reward
            gen_entities = self.extract_clinical_entities(generated_caption)
            
            # Penalize hallucinated pathologies if we have detection results
            if detected_pathologies and i < len(detected_pathologies):
                detected = detected_pathologies[i]
                mentioned_pathologies = gen_entities['pathology']
                
                # Reward correct mentions
                correct = len(mentioned_pathologies & detected)
                # Penalize hallucinations
                hallucinated = len(mentioned_pathologies - detected)
                
                if len(mentioned_pathologies) > 0:
                    entity_reward = (correct - 2.0 * hallucinated) / len(mentioned_pathologies)
                    rewards[i] += entity_reward
        
        return rewards
