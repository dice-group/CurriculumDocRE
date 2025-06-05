import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np
from tqdm import tqdm

import src.config as config
import src.utils as utils

logger = logging.getLogger(__name__)

class DocREDDataset(Dataset):
    """
    Dataset class for DocRED relation extraction.
    """
    def __init__(self, data, tokenizer, relation2id, max_seq_length=512, curriculum_stage=1):
        self.data = data
        self.tokenizer = tokenizer
        self.relation2id = relation2id
        self.max_seq_length = max_seq_length
        self.curriculum_stage = curriculum_stage
        self.examples = self._preprocess_data()
        
    def _preprocess_data(self):
        examples = []
        
        for doc in tqdm(self.data, desc=f"Preprocessing data for Stage {self.curriculum_stage}"):
            doc_text = " ".join([" ".join(sent) for sent in doc["sents"]])
            entity_pairs = []

            for rel in doc.get("labels", []):
                h_idx = rel["h"]
                t_idx = rel["t"]
                
                if h_idx >= len(doc["vertexSet"]) or t_idx >= len(doc["vertexSet"]):
                    continue

                h_sent_ids = [m["sent_id"] for m in doc["vertexSet"][h_idx]]
                t_sent_ids = [m["sent_id"] for m in doc["vertexSet"][t_idx]]

                if not h_sent_ids or not t_sent_ids:
                    continue

                min_distance = min(abs(h - t) for h in h_sent_ids for t in t_sent_ids)

                if self.curriculum_stage == 1 and min_distance <= config.STAGE_1_MAX_DISTANCE:
                    entity_pairs.append((h_idx, t_idx))
                elif self.curriculum_stage == 2 and config.STAGE_1_MAX_DISTANCE < min_distance <= config.STAGE_2_MAX_DISTANCE:
                    entity_pairs.append((h_idx, t_idx))
                elif self.curriculum_stage == 3 and min_distance > config.STAGE_2_MAX_DISTANCE:
                    entity_pairs.append((h_idx, t_idx))

            for h_idx, t_idx in entity_pairs:
                h_name = doc["vertexSet"][h_idx][0]["name"]
                t_name = doc["vertexSet"][t_idx][0]["name"]
                
                prompt = f"What is the relation between {h_name} and {t_name}?"
                input_text = prompt + " " + doc_text

                encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )

                label = torch.zeros(len(self.relation2id))
                confidence = 0.0

                found = False
                for rel in doc.get("labels", []):
                    if rel["h"] == h_idx and rel["t"] == t_idx:
                        r = rel["r"]
                        if r not in self.relation2id:
                            logger.warning(f"❗ Unknown relation '{r}' in doc '{doc.get('title', '')}'. Using 'NA'.")
                            r = "NA"

                        rel_id = self.relation2id.get(r)
                        if rel_id is not None:
                            label[rel_id] = 1.0
                            found = True

                            if "quality_info" in doc:
                                for rel_conf in doc["quality_info"].get("relation_confidences", []):
                                    if rel_conf["h_idx"] == h_idx and rel_conf["t_idx"] == t_idx and rel_conf["relation"] == r:
                                        confidence = rel_conf["confidence"]
                                        break
                        else:
                            logger.warning(f"⚠️ Relation '{r}' not found in relation2id and no fallback.")
                
                if not found:
                    logger.info(f"⚠️ No matching label found for pair {h_idx}-{t_idx} in doc {doc.get('title', '')}. Assigning 'NA'.")
                    na_id = self.relation2id.get("NA", len(self.relation2id))
                    if na_id >= len(label):
                        label = torch.cat([label, torch.tensor([1.0])])
                    else:
                        label[na_id] = 1.0
                    if confidence == 0.0:
                        confidence = 0.3
                

                if confidence == 0.0:
                    confidence = 0.9 if self.curriculum_stage == 1 else 0.7 if self.curriculum_stage == 2 else 0.5

                example = {
                    "input_ids": encoding["input_ids"][0],
                    "attention_mask": encoding["attention_mask"][0],
                    "label": label,
                    "confidence": torch.tensor([confidence], dtype=torch.float),
                    "h_idx": h_idx,
                    "t_idx": t_idx,
                    "doc_id": doc.get("id", ""),
                    "doc_title": doc.get("title", "")
                }

                examples.append(example)

        logger.info(f"Created {len(examples)} examples for curriculum stage {self.curriculum_stage}")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    confidences = torch.stack([item["confidence"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "confidence": confidences,
        "h_idx": [item["h_idx"] for item in batch],
        "t_idx": [item["t_idx"] for item in batch],
        "doc_id": [item["doc_id"] for item in batch],
        "doc_title": [item["doc_title"] for item in batch]
    }

def get_dataloaders(tokenizer, curriculum_stage, batch_size, relation2id):
    try:
        train_data = utils.load_json(config.DOCRED_TRAIN_FILE)
        dev_data = utils.load_json(config.DOCRED_DEV_FILE)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

    train_dataset = DocREDDataset(train_data, tokenizer, relation2id, config.MAX_SEQ_LENGTH, curriculum_stage)
    dev_dataset = DocREDDataset(dev_data, tokenizer, relation2id, config.MAX_SEQ_LENGTH, curriculum_stage)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader

def main():
    from transformers import BertTokenizerFast
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained(config.BASE_MODEL_NAME)
    relation2id = utils.build_relation2id(config.DOCRED_REL_INFO_FILE)

    for stage in [1, 2, 3]:
        train_loader, dev_loader = get_dataloaders(
            tokenizer=tokenizer,
            curriculum_stage=stage,
            batch_size=config.BATCH_SIZE,
            relation2id=relation2id
        )

        if train_loader and dev_loader:
            logger.info(f"Stage {stage}:")
            logger.info(f"  Train examples: {len(train_loader.dataset)}")
            logger.info(f"  Dev examples: {len(dev_loader.dataset)}")
            for batch in train_loader:
                logger.info(f"  Sample batch size: {batch['input_ids'].size()}")
                logger.info(f"  Sample label size: {batch['label'].size()}")
                break

if __name__ == "__main__":
    main()
