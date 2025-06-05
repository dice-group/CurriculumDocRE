"""
Enhanced data quality module with proper integration for CurriculumDocRE.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import logging
from collections import defaultdict
import src.config as config
import src.utils as utils

logger = logging.getLogger(__name__)

class AnnotationConsistencyChecker:
    """
    Checks for consistency in relation annotations across the dataset.
    """
    def __init__(self, relation2id):
        self.relation2id = relation2id
        self.entity_pair_relations = defaultdict(list)
        self.relation_stats = defaultdict(int)
        
    def process_document(self, doc):
        """Process a document to gather entity pair relation statistics."""
        for rel in doc.get("labels", []):
            h_idx, t_idx, r = rel["h"], rel["t"], rel["r"]
            h_type = doc["vertexSet"][h_idx][0].get("type", "UNK")
            t_type = doc["vertexSet"][t_idx][0].get("type", "UNK")
            
           
            pair_type_key = f"{h_type}:{t_type}"
            self.entity_pair_relations[pair_type_key].append(r)
            self.relation_stats[r] += 1
    
    def get_relation_probabilities(self, h_type, t_type):
        """Get probability distribution of relations for a given entity pair type."""
        pair_type_key = f"{h_type}:{t_type}"
        relations = self.entity_pair_relations.get(pair_type_key, [])
        
        if not relations:
            return {}
            
        rel_counts = defaultdict(int)
        for r in relations:
            rel_counts[r] += 1
            
        total = len(relations)
        rel_probs = {r: count/total for r, count in rel_counts.items()}
        
        return rel_probs
    
    def check_annotation_consistency(self, doc):
        """
        Check if annotations in a document are consistent with dataset patterns.
        Returns a list of potentially inconsistent annotations.
        """
        inconsistencies = []
        
        for rel in doc.get("labels", []):
            h_idx, t_idx, r = rel["h"], rel["t"], rel["r"]
            h_type = doc["vertexSet"][h_idx][0].get("type", "UNK")
            t_type = doc["vertexSet"][t_idx][0].get("type", "UNK")
            
            rel_probs = self.get_relation_probabilities(h_type, t_type)
            
            if r in rel_probs and rel_probs[r] < 0.05:  # Less than 5% probability
                inconsistencies.append({
                    "h_idx": h_idx,
                    "t_idx": t_idx,
                    "relation": r,
                    "probability": rel_probs[r],
                    "h_type": h_type,
                    "t_type": t_type
                })
                
        return inconsistencies

class ConfidenceScorer:
    """
    Assigns confidence scores to relation instances based on various features.
    """
    def __init__(self):
        self.features = {
            "sentence_distance": self._sentence_distance_score,
            "entity_frequency": self._entity_frequency_score,
            "relation_frequency": self._relation_frequency_score
        }
        self.entity_counts = defaultdict(int)
        self.relation_counts = defaultdict(int)
        
    def _sentence_distance_score(self, doc, h_idx, t_idx):
        """Score based on sentence distance between entities."""
        h_sent = doc["vertexSet"][h_idx][0]["sent_id"]
        t_sent = doc["vertexSet"][t_idx][0]["sent_id"]
        distance = abs(h_sent - t_sent)
        return 1.0 / (1.0 + distance)
    
    def _entity_frequency_score(self, doc, h_idx, t_idx):
        """Score based on entity frequency in the dataset."""
        h_name = doc["vertexSet"][h_idx][0]["name"]
        t_name = doc["vertexSet"][t_idx][0]["name"]
        
        h_count = self.entity_counts.get(h_name, 1)
        t_count = self.entity_counts.get(t_name, 1)
        
        h_score = min(h_count / 10, 1.0)  # Cap at 1.0
        t_score = min(t_count / 10, 1.0)  # Cap at 1.0
        
        return (h_score + t_score) / 2
    
    def _relation_frequency_score(self, doc, h_idx, t_idx, relation):
        """Score based on relation frequency in the dataset."""
        rel_count = self.relation_counts.get(relation, 1)
        
        if rel_count < 5:
            return 0.5  
        elif rel_count > 100:
            return 0.8  
        else:
            return 0.9  
    
    def process_dataset(self, data):
        """Process dataset to gather entity and relation statistics."""
        for doc in data:
            # Count entities
            for entity in doc["vertexSet"]:
                for mention in entity:
                    self.entity_counts[mention["name"]] += 1
            
            # Count relations
            for rel in doc.get("labels", []):
                self.relation_counts[rel["r"]] += 1
    
    def compute_confidence(self, doc, h_idx, t_idx, relation=None):
        """
        Compute overall confidence score for a relation instance.
        
        Args:
            doc: Document containing the relation
            h_idx: Head entity index
            t_idx: Tail entity index
            relation: Optional relation type
            
        Returns:
            Confidence score between 0 and 1
        """
        scores = []
        
        scores.append(self.features["sentence_distance"](doc, h_idx, t_idx))
        
        scores.append(self.features["entity_frequency"](doc, h_idx, t_idx))
        
        if relation:
            scores.append(self.features["relation_frequency"](doc, h_idx, t_idx, relation))
        
        weights = [0.5, 0.3, 0.2] if relation else [0.6, 0.4]
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_score

class ActiveLearningSelector:
    """
    Selects uncertain instances for expert review using active learning.
    """
    def __init__(self, confidence_scorer):
        self.confidence_scorer = confidence_scorer
        
    def select_uncertain_instances(self, data, threshold=0.5, max_instances=100):
        """
        Select instances with low confidence for expert review.
        
        Args:
            data: List of documents
            threshold: Confidence threshold below which to select instances
            max_instances: Maximum number of instances to select
            
        Returns:
            List of selected instances for review
        """
        uncertain_instances = []
        
        for doc_idx, doc in enumerate(data):
            for rel_idx, rel in enumerate(doc.get("labels", [])):
                h_idx, t_idx, r = rel["h"], rel["t"], rel["r"]
                
                # Compute confidence
                confidence = self.confidence_scorer.compute_confidence(doc, h_idx, t_idx, r)
                if confidence < threshold:
                    uncertain_instances.append({
                        "doc_idx": doc_idx,
                        "rel_idx": rel_idx,
                        "h_idx": h_idx,
                        "t_idx": t_idx,
                        "relation": r,
                        "confidence": confidence,
                        "doc_title": doc["title"]
                    })
        
        uncertain_instances.sort(key=lambda x: x["confidence"])
        return uncertain_instances[:max_instances]

def enhance_data_quality(data, relation2id):
    """
    Main function to enhance data quality through consistency checks,
    confidence scoring, and active learning selection.
    
    Args:
        data: List of documents
        relation2id: Mapping from relation names to IDs
        
    Returns:
        Enhanced data with quality scores
    """
    logger.info("Enhancing data quality...")

    consistency_checker = AnnotationConsistencyChecker(relation2id)
    confidence_scorer = ConfidenceScorer()
    active_learning = ActiveLearningSelector(confidence_scorer)

    logger.info("Processing dataset for consistency patterns...")
    for doc in tqdm(data):
        consistency_checker.process_document(doc)
    
    confidence_scorer.process_dataset(data)
    enhanced_data = []
    for doc in tqdm(data, desc="Enhancing data quality"):
        inconsistencies = consistency_checker.check_annotation_consistency(doc)
        enhanced_doc = doc.copy()
        enhanced_doc["quality_info"] = {
            "inconsistencies": inconsistencies,
            "relation_confidences": []
        }
        for rel in doc.get("labels", []):
            h_idx, t_idx, r = rel["h"], rel["t"], rel["r"]
            confidence = confidence_scorer.compute_confidence(doc, h_idx, t_idx, r)
            
            enhanced_doc["quality_info"]["relation_confidences"].append({
                "h_idx": h_idx,
                "t_idx": t_idx,
                "relation": r,
                "confidence": confidence
            })
        
        enhanced_data.append(enhanced_doc)

    uncertain_instances = active_learning.select_uncertain_instances(enhanced_data)
    logger.info(f"Selected {len(uncertain_instances)} uncertain instances for review")
    
    return enhanced_data, uncertain_instances

def filter_low_quality_instances(data, confidence_threshold=0.3):
    """
    Filter out low-quality instances based on confidence scores.
    
    Args:
        data: Enhanced data with quality information
        confidence_threshold: Threshold below which to filter instances
        
    Returns:
        Filtered data
    """
    filtered_data = []
    
    for doc in data:
        if "quality_info" not in doc:
            filtered_data.append(doc)
            continue
            
        low_conf_relations = set()
        for rel_conf in doc["quality_info"]["relation_confidences"]:
            if rel_conf["confidence"] < confidence_threshold:
                key = (rel_conf["h_idx"], rel_conf["t_idx"], rel_conf["relation"])
                low_conf_relations.add(key)
        
        if "labels" in doc:
            filtered_labels = []
            for rel in doc["labels"]:
                key = (rel["h"], rel["t"], rel["r"])
                if key not in low_conf_relations:
                    filtered_labels.append(rel)
            
            filtered_doc = doc.copy()
            filtered_doc["labels"] = filtered_labels
            filtered_data.append(filtered_doc)
        else:
            filtered_data.append(doc)
    
    return filtered_data

def main():
    """Main function to demonstrate data quality enhancement."""
    relation2id = utils.build_relation2id(config.DOCRED_REL_INFO_FILE)
    
    train_data = utils.load_json(config.DOCRED_TRAIN_FILE)
    
    enhanced_data, uncertain_instances = enhance_data_quality(train_data, relation2id)
    
    filtered_data = filter_low_quality_instances(enhanced_data)
    utils.create_directories()
    
    enhanced_file = os.path.join(config.OUTPUT_DIR, "train_enhanced.json")
    with open(enhanced_file, "w") as f:
        json.dump(enhanced_data, f)
    
    filtered_file = os.path.join(config.OUTPUT_DIR, "train_filtered.json")
    with open(filtered_file, "w") as f:
        json.dump(filtered_data, f)
    
    uncertain_file = os.path.join(config.OUTPUT_DIR, "uncertain_instances.json")
    with open(uncertain_file, "w") as f:
        json.dump(uncertain_instances, f)
    
    logger.info(f"Enhanced data saved to {enhanced_file}")
    logger.info(f"Filtered data saved to {filtered_file}")
    logger.info(f"Uncertain instances saved to {uncertain_file}")
    
    # Print statistics
    logger.info(f"Original instances: {sum(len(doc.get('labels', [])) for doc in train_data)}")
    logger.info(f"Filtered instances: {sum(len(doc.get('labels', [])) for doc in filtered_data)}")
    logger.info(f"Uncertain instances: {len(uncertain_instances)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
