"""
Common utility functions for the CurriculumDocRE project.
"""

import os
import json
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def load_json(file_path):
    """Load JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise

def save_json(data, file_path):
    """Save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise

def build_relation2id(rel_info_file):
    """
    Build relation to ID mapping from relation info file.
    Supports dict or list format.
    """
    logger.info(f"Building relation2id from {os.path.basename(rel_info_file)}...")
    rel_info = load_json(rel_info_file)
    
   
    if isinstance(rel_info, dict):
        relation2id = {rel_id: idx for idx, rel_id in enumerate(rel_info.keys())}
    elif isinstance(rel_info, list) and all(isinstance(item, dict) for item in rel_info):
        if "relation" in rel_info[0]:
            relation2id = {rel["relation"]: idx for idx, rel in enumerate(rel_info)}
        elif "id" in rel_info[0]:
            relation2id = {rel["id"]: idx for idx, rel in enumerate(rel_info)}
        else:
            raise ValueError("Missing 'relation' or 'id' key in list format.")
    elif isinstance(rel_info, list) and all(isinstance(item, str) for item in rel_info):
        relation2id = {rel: idx for idx, rel in enumerate(rel_info)}
    else:
        logger.error(f"Unsupported format in {rel_info_file}. Expected dict or list.")
        raise ValueError(f"Unsupported format in {rel_info_file}")
    
    if "NA" not in relation2id:
        relation2id["NA"] = len(relation2id)
    
    logger.info(f"Loaded {len(relation2id)} relation types (including NA).")
    return relation2id



def get_entity_pairs(doc):
    """Get all entity pairs from a document."""
    entity_pairs = []
    for h_idx, h in enumerate(doc["vertexSet"]):
        for t_idx, t in enumerate(doc["vertexSet"]):
            if h_idx != t_idx: 
                entity_pairs.append((h_idx, t_idx))
    return entity_pairs

def get_relation_distribution(data, relation2id):
    """Get distribution of relations in the dataset."""
    relation_counts = {rel: 0 for rel in relation2id.keys()}
    
    for doc in data:
        for rel in doc.get("labels", []):
            relation_counts[rel["r"]] = relation_counts.get(rel["r"], 0) + 1
    
    return relation_counts

def get_entity_distance(doc, h_idx, t_idx):
    """
    Calculate minimum sentence distance between head and tail entities.
    
    Args:
        doc: Document containing entities
        h_idx: Head entity index
        t_idx: Tail entity index
        
    Returns:
        Minimum sentence distance between any mentions of the entities
    """
    h_sent_ids = [mention["sent_id"] for mention in doc["vertexSet"][h_idx]]
    t_sent_ids = [mention["sent_id"] for mention in doc["vertexSet"][t_idx]]
    
    if not h_sent_ids or not t_sent_ids:
        return float('inf')
    min_distance = min([abs(h - t) for h in h_sent_ids for t in t_sent_ids])
    return min_distance

def get_curriculum_stage(doc, h_idx, t_idx):
    """
    Determine curriculum stage based on entity distance.
    
    Args:
        doc: Document containing entities
        h_idx: Head entity index
        t_idx: Tail entity index
        
    Returns:
        Curriculum stage (1, 2, or 3)
    """
    from src.config import STAGE_1_MAX_DISTANCE, STAGE_2_MAX_DISTANCE
    
    distance = get_entity_distance(doc, h_idx, t_idx)
    
    if distance <= STAGE_1_MAX_DISTANCE:
        return 1  
    elif distance <= STAGE_2_MAX_DISTANCE:
        return 2  
    else:
        return 3  

def create_directories():
    """Create necessary directories for output."""
    from src.config import OUTPUT_DIR, RESULTS_SAVE_DIR, MODEL_SAVE_DIR
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Created output directories")
