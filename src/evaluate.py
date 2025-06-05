"""
Efficient evaluation script for CurriculumDocRE with memory optimization.
"""

import os
import json
import torch
import logging
from tqdm import tqdm
from transformers import BertTokenizerFast
import numpy as np
import src.config as config
import src.utils as utils
from src.model import EnhancedCurriculumDocREModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def process_document_in_chunks(doc, tokenizer, model, relation2id, device, chunk_size=2):
    doc_text = " ".join([" ".join(sent) for sent in doc["sents"]])

    entity_pairs = [(h_idx, t_idx) for h_idx in range(len(doc["vertexSet"]))
                    for t_idx in range(len(doc["vertexSet"])) if h_idx != t_idx]

    all_predictions = []

    for i in range(0, len(entity_pairs), chunk_size):
        chunk_pairs = entity_pairs[i:i + chunk_size]
        batch_inputs = []

        for h_idx, t_idx in chunk_pairs:
            h_name = doc["vertexSet"][h_idx][0]["name"]
            t_name = doc["vertexSet"][t_idx][0]["name"]
            prompt = f"What is the relation between {h_name} and {t_name}?"
            input_text = prompt + " " + doc_text

            encoding = tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            )

            batch_inputs.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "h_idx": h_idx,
                "t_idx": t_idx
            })

        input_ids = torch.stack([item["input_ids"] for item in batch_inputs]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch_inputs]).to(device)

        stages = [utils.get_curriculum_stage(doc, item["h_idx"], item["t_idx"]) for item in batch_inputs]

        for idx, stage in enumerate(stages):
            single_input_ids = input_ids[idx:idx + 1]
            single_attention_mask = attention_mask[idx:idx + 1]

            with torch.no_grad():
                relation_logits, confidence_logit = model(
                    input_ids=single_input_ids,
                    attention_mask=single_attention_mask,
                    current_curriculum_stage=stage
                )

            relation_probs = torch.sigmoid(relation_logits).cpu().numpy()[0]
            confidence = torch.sigmoid(confidence_logit).cpu().numpy()[0][0]

        
            prob_threshold = config.PREDICTION_THRESHOLD.get(stage, 0.05)
            confidence_threshold = 0.62
            min_prob_required = 0.4

        
            rel_id = np.argmax(relation_probs)
            prob = relation_probs[rel_id]

            rel_name = next((name for name, idx_val in relation2id.items() if idx_val == rel_id), None)

            if (
                prob > prob_threshold and
                confidence > confidence_threshold and
                prob >= min_prob_required and
                rel_name and rel_name != "NA"
            ):
                h_idx = batch_inputs[idx]["h_idx"]
                t_idx = batch_inputs[idx]["t_idx"]
                all_predictions.append({
                    "h": h_idx,
                    "t": t_idx,
                    "r": rel_name,
                    "probability": float(prob),
                    "confidence": float(confidence)
                })

        del input_ids, attention_mask
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # (Optional) Limit predictions per document
    MAX_RELATIONS_PER_DOC = 50
    if len(all_predictions) > MAX_RELATIONS_PER_DOC:
        all_predictions = sorted(all_predictions, key=lambda x: -x["probability"])[:MAX_RELATIONS_PER_DOC]

    logger.info(f"Predicted {len(all_predictions)} relations for doc '{doc.get('title')}' with {len(doc['vertexSet'])} entities.")
    return all_predictions


def process_document_in_chunks_old(doc, tokenizer, model, relation2id, device, chunk_size=2):
    doc_text = " ".join([" ".join(sent) for sent in doc["sents"]])

    entity_pairs = [(h_idx, t_idx) for h_idx in range(len(doc["vertexSet"]))
                    for t_idx in range(len(doc["vertexSet"])) if h_idx != t_idx]

    all_predictions = []

    for i in range(0, len(entity_pairs), chunk_size):
        chunk_pairs = entity_pairs[i:i + chunk_size]
        batch_inputs = []

        for h_idx, t_idx in chunk_pairs:
            h_name = doc["vertexSet"][h_idx][0]["name"]
            t_name = doc["vertexSet"][t_idx][0]["name"]
            prompt = f"What is the relation between {h_name} and {t_name}?"
            input_text = prompt + " " + doc_text

            encoding = tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            )

            batch_inputs.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "h_idx": h_idx,
                "t_idx": t_idx
            })

        input_ids = torch.stack([item["input_ids"] for item in batch_inputs]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch_inputs]).to(device)

        stages = [utils.get_curriculum_stage(doc, item["h_idx"], item["t_idx"]) for item in batch_inputs]

        for idx, stage in enumerate(stages):
            single_input_ids = input_ids[idx:idx + 1]
            single_attention_mask = attention_mask[idx:idx + 1]

            with torch.no_grad():
                relation_logits, confidence_logit = model(
                    input_ids=single_input_ids,
                    attention_mask=single_attention_mask,
                    current_curriculum_stage=stage
                )

            relation_probs = torch.sigmoid(relation_logits).cpu().numpy()[0]
            confidence = torch.sigmoid(confidence_logit).cpu().numpy()[0][0]

            threshold = config.PREDICTION_THRESHOLD.get(stage, 0.9)
            top_k = 1
            top_indices = np.argsort(-relation_probs)[:top_k]

            for rel_id in top_indices:
                prob = relation_probs[rel_id]
                rel_name = None
                for name, idx_val in relation2id.items():
                    if idx_val == rel_id:
                        rel_name = name
                        break

                if prob > threshold and rel_name and rel_name != "NA":
                    h_idx = batch_inputs[idx]["h_idx"]
                    t_idx = batch_inputs[idx]["t_idx"]
                    all_predictions.append({
                        "h": h_idx,
                        "t": t_idx,
                        "r": rel_name,
                        "probability": float(prob),
                        "confidence": float(confidence)
                    })

        del input_ids, attention_mask
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"Predicted {len(all_predictions)} relations for doc '{doc.get('title')}' with {len(doc['vertexSet'])} entities.")
    return all_predictions

def evaluate_dataset(data, tokenizer, model, relation2id, device):
    all_predictions = {}
    for doc in tqdm(data, desc="Evaluating documents"):
        doc_id = doc.get("id", doc.get("title", ""))
        predictions = process_document_in_chunks(doc, tokenizer, model, relation2id, device)
        all_predictions[doc_id] = predictions
    return all_predictions

def format_predictions_for_docred(predictions):
    docred_predictions = []
    for doc_id, doc_preds in predictions.items():
        for pred in doc_preds:
            docred_predictions.append({
                "title": doc_id,
                "h_idx": pred["h"],
                "t_idx": pred["t"],
                "r": pred["r"]
            })
    return docred_predictions

def main():
    utils.set_seed(config.SEED)
    utils.create_directories()

    relation2id = utils.build_relation2id(config.DOCRED_REL_INFO_FILE)
    dev_data = utils.load_json(config.DOCRED_DEV_FILE)

    tokenizer = BertTokenizerFast.from_pretrained(config.BASE_MODEL_NAME)
    model = EnhancedCurriculumDocREModel(
        base_model_name=config.BASE_MODEL_NAME,
        num_relations=len(relation2id)
    )

    if os.path.exists(config.MODEL_SAVE_PATH):
        logger.info(f"Loading model from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location="cpu")
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
    else:
        logger.warning(f"Model file {config.MODEL_SAVE_PATH} not found. Using untrained model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    predictions = evaluate_dataset(dev_data, tokenizer, model, relation2id, device)
    docred_predictions = format_predictions_for_docred(predictions)

    predictions_file = os.path.join(config.RESULTS_SAVE_DIR, "predictions.json")
    with open(predictions_file, "w") as f:
        json.dump(docred_predictions, f, indent=2)

    logger.info(f"Predictions saved to {predictions_file}")
    logger.info(f"Total predictions: {len(docred_predictions)}")

    logger.info("\nTo evaluate using the official DocRED script:")
    logger.info("1. Clone the DocRED repository:")
    logger.info("   git clone https://github.com/thunlp/DocRED.git")
    logger.info("2. Run the evaluation script:")
    logger.info(f"   cd DocRED/code && python eval.py -g ../../{config.DOCRED_DEV_FILE} -p ../../{predictions_file}")

if __name__ == "__main__":
    main()
