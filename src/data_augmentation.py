"""
Enhanced data augmentation module with proper integration for CurriculumDocRE.
"""

import os
import json
import random
import copy
import logging
from tqdm import tqdm
from collections import defaultdict
import src.config as config
import src.utils as utils

logger = logging.getLogger(__name__)

class EntitySubstitution:
    """
    Substitutes entities with similar entities to create augmented training examples.
    """
    def __init__(self, data):
        self.entity_by_type = defaultdict(list)
        self.build_entity_index(data)
        
    def build_entity_index(self, data):
        """Build an index of entities by type."""
        for doc in data:
            for entity_idx, entity in enumerate(doc["vertexSet"]):
                entity_type = entity[0].get("type", "UNK")
                entity_name = entity[0]["name"]
                
                self.entity_by_type[entity_type].append({
                    "name": entity_name,
                    "mentions": entity,
                    "doc_title": doc["title"],
                    "entity_idx": entity_idx
                })
    
    def get_similar_entity(self, entity_type, exclude_name=None):
        """Get a random entity of the same type, excluding the given name."""
        candidates = [e for e in self.entity_by_type.get(entity_type, []) 
                     if e["name"] != exclude_name]
        
        if not candidates:
            return None
            
        return random.choice(candidates)
    
    def substitute_entity(self, doc, entity_idx, substitution_prob=0.5):
        """
        Substitute an entity with a similar one with some probability.
        
        Args:
            doc: Document containing the entity
            entity_idx: Index of the entity to potentially substitute
            substitution_prob: Probability of performing substitution
            
        Returns:
            Tuple of (substituted_doc, substitution_info) or (None, None) if no substitution
        """
        if random.random() > substitution_prob:
            return None, None
            
        if entity_idx >= len(doc["vertexSet"]):
            return None, None
            
        entity = doc["vertexSet"][entity_idx]
        entity_type = entity[0].get("type", "UNK")
        entity_name = entity[0]["name"]
        similar_entity = self.get_similar_entity(entity_type, entity_name)
        if not similar_entity:
            return None, None
            
        substituted_doc = copy.deepcopy(doc)
        
        for mention_idx, mention in enumerate(entity):
          
            sent_id = mention["sent_id"]
            if sent_id < len(substituted_doc["sents"]):
                for i, word in enumerate(substituted_doc["sents"][sent_id]):
                    if word == entity_name:
                        substituted_doc["sents"][sent_id][i] = similar_entity["name"]
            
            if mention_idx < len(substituted_doc["vertexSet"][entity_idx]):
                substituted_doc["vertexSet"][entity_idx][mention_idx]["name"] = similar_entity["name"]
        
        substitution_info = {
            "entity_idx": entity_idx,
            "original_name": entity_name,
            "substituted_name": similar_entity["name"],
            "entity_type": entity_type
        }
        
        return substituted_doc, substitution_info

class EvidenceMasking:
    """
    Masks evidence sentences to create more challenging training examples.
    """
    def __init__(self):
        pass
    
    def identify_evidence_sentences(self, doc, h_idx, t_idx):
        """
        Identify sentences that likely contain evidence for a relation.
        
        Args:
            doc: Document containing the relation
            h_idx: Head entity index
            t_idx: Tail entity index
            
        Returns:
            List of sentence IDs that likely contain evidence
        """
        evidence_sent_ids = set()
        
        for mention in doc["vertexSet"][h_idx]:
            evidence_sent_ids.add(mention["sent_id"])
        
        for mention in doc["vertexSet"][t_idx]:
            evidence_sent_ids.add(mention["sent_id"])
            
        h_sent_ids = [mention["sent_id"] for mention in doc["vertexSet"][h_idx]]
        t_sent_ids = [mention["sent_id"] for mention in doc["vertexSet"][t_idx]]
        
        if h_sent_ids and t_sent_ids:
            min_h_sent = min(h_sent_ids)
            max_h_sent = max(h_sent_ids)
            min_t_sent = min(t_sent_ids)
            max_t_sent = max(t_sent_ids)
            
           
            for sent_id in range(min(min_h_sent, min_t_sent), max(max_h_sent, max_t_sent) + 1):
                evidence_sent_ids.add(sent_id)
        
        return sorted(list(evidence_sent_ids))
    
    def mask_evidence(self, doc, relation, mask_prob=0.3):
        """
        Mask evidence sentences for a relation with some probability.
        
        Args:
            doc: Document containing the relation
            relation: Relation to mask evidence for
            mask_prob: Probability of masking each evidence sentence
            
        Returns:
            Tuple of (masked_doc, masking_info) or (None, None) if no masking
        """
        h_idx, t_idx = relation["h"], relation["t"]
        
 
        evidence_sent_ids = self.identify_evidence_sentences(doc, h_idx, t_idx)
   
        sentences_to_mask = []
        for sent_id in evidence_sent_ids:
            if random.random() < mask_prob:
                sentences_to_mask.append(sent_id)
  
        if not sentences_to_mask:
            return None, None
 
        masked_doc = copy.deepcopy(doc)

        for sent_id in sentences_to_mask:
            if sent_id < len(masked_doc["sents"]):
               
                entity_words = set()
  
                for entity_idx in [h_idx, t_idx]:
                    for mention in masked_doc["vertexSet"][entity_idx]:
                        if mention["sent_id"] == sent_id:
                            entity_words.add(mention["name"])
                
                for i, word in enumerate(masked_doc["sents"][sent_id]):
                    if word not in entity_words:
                        masked_doc["sents"][sent_id][i] = "[MASK]"
  
        masking_info = {
            "relation": relation,
            "masked_sent_ids": sentences_to_mask,
            "evidence_sent_ids": evidence_sent_ids
        }
        
        return masked_doc, masking_info

class CrossDocumentRelationTransfer:
    """
    Transfers relations between documents to create new training examples.
    """
    def __init__(self, data):
        self.relation_templates = defaultdict(list)
        self.build_relation_templates(data)
        
    def build_relation_templates(self, data):
        """Build templates of relations by entity types."""
        for doc in data:
            for rel in doc.get("labels", []):
                h_idx, t_idx, r = rel["h"], rel["t"], rel["r"]

                if h_idx >= len(doc["vertexSet"]) or t_idx >= len(doc["vertexSet"]):
                    continue
                    
                h_type = doc["vertexSet"][h_idx][0].get("type", "UNK")
                t_type = doc["vertexSet"][t_idx][0].get("type", "UNK")

                template_key = f"{h_type}:{t_type}:{r}"
 
                self.relation_templates[template_key].append({
                    "doc_title": doc["title"],
                    "head_idx": h_idx,
                    "tail_idx": t_idx,
                    "relation": r,
                    "head_type": h_type,
                    "tail_type": t_type
                })
    
    def find_compatible_entities(self, doc, template_key):
        """
        Find entity pairs in the document that match the template entity types.
        
        Args:
            doc: Document to search for compatible entities
            template_key: Template key in format "head_type:tail_type:relation"
            
        Returns:
            List of compatible entity pairs (head_idx, tail_idx)
        """
        if ":" not in template_key:
            return []
            
        h_type, t_type, _ = template_key.split(":")
        compatible_pairs = []

        h_candidates = []
        t_candidates = []
        
        for entity_idx, entity in enumerate(doc["vertexSet"]):
            entity_type = entity[0].get("type", "UNK")
            
            if entity_type == h_type:
                h_candidates.append(entity_idx)
            
            if entity_type == t_type:
                t_candidates.append(entity_idx)

        for h_idx in h_candidates:
            for t_idx in t_candidates:
                if h_idx != t_idx:  # Avoid self-relations
                    compatible_pairs.append((h_idx, t_idx))
        
        return compatible_pairs
    
    def transfer_relation(self, doc, transfer_prob=0.2):
        """
        Transfer a relation from another document to this one.
        
        Args:
            doc: Document to potentially add a relation to
            transfer_prob: Probability of transferring a relation
            
        Returns:
            Tuple of (augmented_doc, transfer_info) or (None, None) if no transfer
        """

        if random.random() > transfer_prob:
            return None, None
  
        if not self.relation_templates:
            return None, None
            
        template_key = random.choice(list(self.relation_templates.keys()))
        templates = self.relation_templates[template_key]
        
        if not templates:
            return None, None
            
        template = random.choice(templates)
  
        compatible_pairs = self.find_compatible_entities(doc, template_key)
        
        if not compatible_pairs:
            return None, None
    
        h_idx, t_idx = random.choice(compatible_pairs)
   
        relation_exists = False
        for rel in doc.get("labels", []):
            if rel["h"] == h_idx and rel["t"] == t_idx and rel["r"] == template["relation"]:
                relation_exists = True
                break
                
        if relation_exists:
            return None, None
   
        augmented_doc = copy.deepcopy(doc)
    
        if "labels" not in augmented_doc:
            augmented_doc["labels"] = []
            
        new_relation = {
            "h": h_idx,
            "t": t_idx,
            "r": template["relation"]
        }
        
        augmented_doc["labels"].append(new_relation)
    
        transfer_info = {
            "template_key": template_key,
            "source_doc": template["doc_title"],
            "target_doc": doc["title"],
            "relation": template["relation"],
            "head_idx": h_idx,
            "tail_idx": t_idx
        }
        
        return augmented_doc, transfer_info

def augment_data(data, augmentation_factor=2):
    """
    Main function to augment data through entity substitution,
    evidence masking, and cross-document relation transfer.
    
    Args:
        data: List of documents
        augmentation_factor: Target factor for data augmentation
        
    Returns:
        Augmented data
    """
    logger.info("Augmenting data...")
  
    entity_substitution = EntitySubstitution(data)
    evidence_masking = EvidenceMasking()
    relation_transfer = CrossDocumentRelationTransfer(data)
  
    stats = {
        "entity_substitutions": 0,
        "evidence_maskings": 0,
        "relation_transfers": 0
    }
  
    augmented_data = copy.deepcopy(data)
    logger.info("Applying entity substitution...")
    entity_substituted_docs = []
    
    for doc in tqdm(data, desc="Entity substitution"):
        for entity_idx in range(len(doc.get("vertexSet", []))):
            substituted_doc, sub_info = entity_substitution.substitute_entity(
                doc, entity_idx, substitution_prob=config.ENTITY_SUBSTITUTION_PROB
            )
            
            if substituted_doc:
                entity_substituted_docs.append(substituted_doc)
                stats["entity_substitutions"] += 1

    logger.info("Applying evidence masking...")
    evidence_masked_docs = []
    
    for doc in tqdm(data, desc="Evidence masking"):
        for rel in doc.get("labels", []):
            masked_doc, mask_info = evidence_masking.mask_evidence(
                doc, rel, mask_prob=config.EVIDENCE_MASKING_PROB
            )
            
            if masked_doc:
                evidence_masked_docs.append(masked_doc)
                stats["evidence_maskings"] += 1
  
    logger.info("Applying relation transfer...")
    relation_transferred_docs = []
    
    for doc in tqdm(data, desc="Relation transfer"):
        transferred_doc, transfer_info = relation_transfer.transfer_relation(
            doc, transfer_prob=config.RELATION_TRANSFER_PROB
        )
        
        if transferred_doc:
            relation_transferred_docs.append(transferred_doc)
            stats["relation_transfers"] += 1
   
    augmented_data.extend(entity_substituted_docs)
    augmented_data.extend(evidence_masked_docs)
    augmented_data.extend(relation_transferred_docs)
  
    logger.info(f"Original documents: {len(data)}")
    logger.info(f"Entity substitutions: {stats['entity_substitutions']}")
    logger.info(f"Evidence maskings: {stats['evidence_maskings']}")
    logger.info(f"Relation transfers: {stats['relation_transfers']}")
    logger.info(f"Total augmented documents: {len(augmented_data)}")
    
    return augmented_data

def main():
    """Main function to demonstrate data augmentation."""
  
    train_data = utils.load_json(config.DOCRED_TRAIN_FILE)
    augmented_data = augment_data(train_data)
  
    utils.create_directories()
    
    augmented_file = os.path.join(config.OUTPUT_DIR, "train_augmented.json")
    with open(augmented_file, "w") as f:
        json.dump(augmented_data, f)
    
    logger.info(f"Augmented data saved to {augmented_file}")
   
    logger.info(f"Original instances: {sum(len(doc.get('labels', [])) for doc in train_data)}")
    logger.info(f"Augmented instances: {sum(len(doc.get('labels', [])) for doc in augmented_data)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
