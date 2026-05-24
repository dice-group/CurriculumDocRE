# data_augmentation.py - Augmentation strategies for rare relations

import copy
import random
from collections import defaultdict


def keep_only_positive_labels(doc):
    """Remove any NA relation labels from the document."""
    if "labels" in doc:
        doc["labels"] = [rel for rel in doc["labels"] if rel.get("r") != "NA"]
    return doc


class EntitySubstitution:
    """
    Substitute an entity mention with another entity of the same type.
    Only modifies documents that contain at least one positive relation.
    """
    def __init__(self, data):
        self.entity_by_type = defaultdict(list)
        self._build_index(data)

    def _build_index(self, data):
        for doc in data:
            for entity_idx, entity in enumerate(doc["vertexSet"]):
                if not entity:
                    continue
                entity_type = entity[0].get("type", "UNK")
                entity_name = entity[0]["name"]
                self.entity_by_type[entity_type].append({
                    "name": entity_name,
                    "doc_title": doc["title"],
                    "entity_idx": entity_idx
                })

    def get_similar_entity(self, entity_type, exclude_name=None):
        candidates = [e for e in self.entity_by_type.get(entity_type, []) if e["name"] != exclude_name]
        if not candidates:
            return None
        return random.choice(candidates)

    def substitute_entity(self, doc, entity_idx, substitution_prob=0.3):
        if random.random() > substitution_prob:
            return None, None
        if entity_idx >= len(doc["vertexSet"]):
            return None, None
        entity = doc["vertexSet"][entity_idx]
        entity_type = entity[0].get("type", "UNK")
        entity_name = entity[0]["name"]
        similar = self.get_similar_entity(entity_type, entity_name)
        if not similar:
            return None, None
        new_doc = copy.deepcopy(doc)
        new_name = similar["name"]
        # Update all mentions: replace name in sentences and vertexSet
        for mention_idx, mention in enumerate(entity):
            sent_id = mention["sent_id"]
            if sent_id < len(new_doc["sents"]):
                for i, word in enumerate(new_doc["sents"][sent_id]):
                    if word == entity_name:
                        new_doc["sents"][sent_id][i] = new_name
            if mention_idx < len(new_doc["vertexSet"][entity_idx]):
                new_doc["vertexSet"][entity_idx][mention_idx]["name"] = new_name
        return new_doc, {"original": entity_name, "new": new_name, "type": entity_type}


class EvidenceMasking:
    """
    Mask non‑entity words in one evidence sentence to force the model to focus on minimal cues.
    """
    def __init__(self):
        pass

    def mask_evidence(self, doc, relation, mask_prob=0.2):
        h, t = relation["h"], relation["t"]
        # Collect sentence IDs where either entity appears
        sent_ids = set()
        for mention in doc["vertexSet"][h]:
            sent_ids.add(mention["sent_id"])
        for mention in doc["vertexSet"][t]:
            sent_ids.add(mention["sent_id"])
        if not sent_ids or random.random() > mask_prob:
            return None, None
        sent_to_mask = random.choice(list(sent_ids))
        new_doc = copy.deepcopy(doc)
        # Gather entity names that appear in this sentence (to keep them unmasked)
        entity_names = set()
        for idx in [h, t]:
            for mention in new_doc["vertexSet"][idx]:
                if mention["sent_id"] == sent_to_mask:
                    entity_names.add(mention["name"])
        # Mask other words
        for i, word in enumerate(new_doc["sents"][sent_to_mask]):
            if word not in entity_names:
                new_doc["sents"][sent_to_mask][i] = "[MASK]"
        return new_doc, {"masked_sent": sent_to_mask, "entities_kept": list(entity_names)}


class CrossDocumentRelationTransfer:
    """
    Transfer a relation from a source document to a target document
    if they contain entities of compatible types.
    """
    def __init__(self, data):
        self.templates = defaultdict(list)
        self._build_templates(data)

    def _build_templates(self, data):
        for doc in data:
            for rel in doc.get("labels", []):
                h, t, r = rel["h"], rel["t"], rel["r"]
                if h >= len(doc["vertexSet"]) or t >= len(doc["vertexSet"]):
                    continue
                h_type = doc["vertexSet"][h][0].get("type", "UNK")
                t_type = doc["vertexSet"][t][0].get("type", "UNK")
                key = (h_type, t_type, r)
                self.templates[key].append({
                    "head_type": h_type,
                    "tail_type": t_type,
                    "relation": r,
                    "source_title": doc["title"]
                })

    def find_compatible_pair(self, doc, template_key):
        h_type, t_type, r = template_key
        h_candidates = []
        t_candidates = []
        for idx, entity in enumerate(doc["vertexSet"]):
            if not entity:
                continue
            e_type = entity[0].get("type", "UNK")
            if e_type == h_type:
                h_candidates.append(idx)
            if e_type == t_type:
                t_candidates.append(idx)
        # Avoid self‑pairs
        for h in h_candidates:
            for t in t_candidates:
                if h != t:
                    return h, t
        return None, None

    def transfer_relation(self, doc, transfer_prob=0.2):
        if random.random() > transfer_prob:
            return None, None
        if not self.templates:
            return None, None
        # Choose a random template
        key = random.choice(list(self.templates.keys()))
        h_type, t_type, r = key
        h, t = self.find_compatible_pair(doc, key)
        if h is None:
            return None, None
        # Avoid duplicate relation for the same pair
        for existing in doc.get("labels", []):
            if existing["h"] == h and existing["t"] == t and existing["r"] == r:
                return None, None
        new_doc = copy.deepcopy(doc)
        if "labels" not in new_doc:
            new_doc["labels"] = []
        new_doc["labels"].append({
            "h": h,
            "t": t,
            "r": r,
            "evidence": []   # placeholder
        })
        return new_doc, {"source_key": key, "target_pair": (h, t), "relation": r}


def augment_data(data, augmentation_factor=1,
                 ent_sub_prob=0.3, evi_mask_prob=0.2, rel_trans_prob=0.2):
    """
    Main function to generate augmented copies of documents that contain positive relations.
    Each augmentation operator may create multiple copies per document.
    Returns a list of augmented documents (each is a deep copy with modified content).
    """
    # Keep only documents that have at least one positive relation
    positive_docs = [doc for doc in data if any(rel.get("r") != "NA" for rel in doc.get("labels", []))]
    if not positive_docs:
        return []

    sub = EntitySubstitution(positive_docs)
    mask = EvidenceMasking()
    trans = CrossDocumentRelationTransfer(positive_docs)

    augmented = []
    for _ in range(augmentation_factor):
        # Entity substitution
        for doc in positive_docs:
            for entity_idx in range(len(doc.get("vertexSet", []))):
                new_doc, _ = sub.substitute_entity(doc, entity_idx, ent_sub_prob)
                if new_doc:
                    new_doc = keep_only_positive_labels(new_doc)
                    augmented.append(new_doc)
        # Evidence masking
        for doc in positive_docs:
            for rel in doc.get("labels", []):
                new_doc, _ = mask.mask_evidence(doc, rel, evi_mask_prob)
                if new_doc:
                    new_doc = keep_only_positive_labels(new_doc)
                    augmented.append(new_doc)
        # Cross‑document relation transfer
        for doc in positive_docs:
            new_doc, _ = trans.transfer_relation(doc, rel_trans_prob)
            if new_doc:
                new_doc = keep_only_positive_labels(new_doc)
                augmented.append(new_doc)

    return augmented
