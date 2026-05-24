# evaluation.py - Official evaluation metrics for DocRED and Re‑DocRED

import os
import json
import numpy as np
from tqdm import tqdm

# -------------------- Helper functions --------------------

def load_relation_mapping(meta_dir="meta"):
    """Load relation ID mapping from rel2id.json."""
    with open(os.path.join(meta_dir, "rel2id.json"), "r") as f:
        rel2id = json.load(f)
    id2rel = {v: k for k, v in rel2id.items()}
    return rel2id, id2rel


def to_official(preds, features, evi_preds=None, scores=None, topks=None):
    """
    Convert model predictions (numpy arrays) to the official evaluation format.
    
    Args:
        preds: numpy array of shape (total_pairs, num_relations) with binary predictions.
        features: list of feature dictionaries (as returned by prepro.read_docred).
        evi_preds: optional numpy array of evidence predictions.
        scores, topks: optional outputs from ATLoss.get_score.
    
    Returns:
        official_results: list of dicts with keys title, h_idx, t_idx, r, (evidence, score).
        all_results: same as official_results but includes all predictions (including NA?).
    """
    # Extract metadata from features
    h_idx_list, t_idx_list, title_list, sent_len_list = [], [], [], []
    for f in features:
        # Support both original and pseudo documents (with entity_map)
        if "entity_map" in f:
            hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
        else:
            hts = f["hts"]
        for h, t in hts:
            h_idx_list.append(h)
            t_idx_list.append(t)
            title_list.append(f["title"])
            sent_len_list.append(len(f["sent_pos"]))
    assert len(h_idx_list) == preds.shape[0], "Mismatch between predictions and features"

    rel2id, id2rel = load_relation_mapping()
    official_res = []
    all_res = []

    for i in range(preds.shape[0]):
        if scores is not None and topks is not None:
            # Extract relative scores (subtract NA score)
            na_score = scores[i][-1] - 1
            if 0 in topks[i]:
                na_idx = np.where(topks[i] == 0)[0][0]
                na_score = scores[i][na_idx]
            rel_scores = scores[i] - na_score
            pred_relations = topks[i]
        else:
            pred_relations = np.nonzero(preds[i])[0].tolist()

        for r_id in pred_relations:
            if r_id == 0:   # NA relation, skip in official results
                continue
            res_item = {
                "title": title_list[i],
                "h_idx": h_idx_list[i],
                "t_idx": t_idx_list[i],
                "r": id2rel[r_id]
            }
            if evi_preds is not None and i < len(evi_preds):
                evi = np.nonzero(evi_preds[i])[0].tolist()
                res_item["evidence"] = [e for e in evi if e < sent_len_list[i]]
            if scores is not None:
                res_item["score"] = float(rel_scores[np.where(topks[i] == r_id)[0][0]])
            official_res.append(res_item)
            all_res.append(res_item)
    return official_res, all_res


def gen_train_facts(data_file_name, truth_dir):
    """
    Generate or load the set of training facts (used for Ign F1 calculation).
    """
    base = os.path.basename(data_file_name)
    fact_file = os.path.join(truth_dir, base.replace(".json", ".fact"))
    if os.path.exists(fact_file):
        with open(fact_file, "r") as f:
            return set(tuple(x) for x in json.load(f))
    # Otherwise compute from original training data
    with open(data_file_name, "r") as f:
        data = json.load(f)
    facts = set()
    for doc in data:
        vertexSet = doc["vertexSet"]
        for label in doc.get("labels", []):
            rel = label["r"]
            h = label["h"]
            t = label["t"]
            for mh in vertexSet[h]:
                for mt in vertexSet[t]:
                    facts.add((mh["name"], mt["name"], rel))
    with open(fact_file, "w") as f:
        json.dump(list(facts), f)
    return facts


def official_evaluate(preds, data_dir, train_file="train_annotated.json", dev_file="dev.json"):
    """
    Official evaluation function for DocRED.
    Returns:
        re_metrics: [precision, recall, f1]
        evi_metrics: [precision, recall, f1]
        ign_metrics: [precision_ign, recall_ign, f1_ign]
        (the fourth return value is unused, kept for compatibility)
    """
    truth_dir = os.path.join(data_dir, "ref")
    os.makedirs(truth_dir, exist_ok=True)

    # Generate training facts
    train_path = os.path.join(data_dir, train_file)
    fact_in_train_annotated = gen_train_facts(train_path, truth_dir)
    train_distant_path = os.path.join(data_dir, "train_distant.json")
    fact_in_train_distant = gen_train_facts(train_distant_path, truth_dir) if os.path.exists(train_distant_path) else set()

    # Load ground truth
    dev_path = os.path.join(data_dir, dev_file)
    with open(dev_path, "r") as f:
        gold_data = json.load(f)

    # Build gold dictionaries
    gold_relations = {}      # (title, r, h, t) -> evidence set
    tot_evidences = 0
    title2vertex = {}
    for doc in gold_data:
        title = doc["title"]
        title2vertex[title] = doc["vertexSet"]
        if "labels" not in doc:
            continue
        for label in doc["labels"]:
            r = label["r"]
            h = label["h"]
            t = label["t"]
            evi = set(label.get("evidence", []))
            gold_relations[(title, r, h, t)] = evi
            tot_evidences += len(evi)

    tot_relations = len(gold_relations)

    # Deduplicate predictions (same title, h, t, r)
    preds_sorted = sorted(preds, key=lambda x: (x["title"], x["h_idx"], x["t_idx"], x["r"]))
    unique_preds = []
    for i, p in enumerate(preds_sorted):
        if i == 0 or (p["title"], p["h_idx"], p["t_idx"], p["r"]) != \
                      (preds_sorted[i-1]["title"], preds_sorted[i-1]["h_idx"], preds_sorted[i-1]["t_idx"], preds_sorted[i-1]["r"]):
            unique_preds.append(p)
    preds = unique_preds

    correct = 0
    correct_evi = 0
    pred_evi_count = 0
    correct_ign = 0   # correct ignoring relations seen in training

    for p in preds:
        title = p["title"]
        h = p["h_idx"]
        t = p["t_idx"]
        r = p["r"]
        evi = set(p.get("evidence", []))
        pred_evi_count += len(evi)

        if (title, r, h, t) in gold_relations:
            correct += 1
            gold_evi = gold_relations[(title, r, h, t)]
            correct_evi += len(evi & gold_evi)

            # Check if relation appears in training
            vertex = title2vertex.get(title, [])
            in_train = False
            if h < len(vertex) and t < len(vertex):
                for mh in vertex[h]:
                    for mt in vertex[t]:
                        if (mh["name"], mt["name"], r) in fact_in_train_annotated:
                            in_train = True
                            break
                    if in_train:
                        break
            if not in_train:
                correct_ign += 1

    # Compute metrics
    precision = correct / len(preds) if len(preds) > 0 else 0
    recall = correct / tot_relations if tot_relations > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    evi_precision = correct_evi / pred_evi_count if pred_evi_count > 0 else 0
    evi_recall = correct_evi / tot_evidences if tot_evidences > 0 else 0
    evi_f1 = 2 * evi_precision * evi_recall / (evi_precision + evi_recall) if (evi_precision + evi_recall) > 0 else 0

    ign_precision = correct_ign / len(preds) if len(preds) > 0 else 0
    ign_recall = correct_ign / tot_relations if tot_relations > 0 else 0
    ign_f1 = 2 * ign_precision * ign_recall / (ign_precision + ign_recall) if (ign_precision + ign_recall) > 0 else 0

    return [precision, recall, f1], [evi_precision, evi_recall, evi_f1], [ign_precision, ign_recall, ign_f1], []
