# prepro.py - Data preprocessing for CurriculumDocRE

from tqdm import tqdm
import ujson as json
import torch
import numpy as np
import os

# Load relation mapping (adjust path if needed)
docred_rel2id = json.load(open('meta/rel2id.json', 'r'))


def get_entity_distance(doc, h_idx, t_idx):
    """Minimum sentence distance between two entities."""
    h_sent_ids = [m["sent_id"] for m in doc["vertexSet"][h_idx] if m.get("sent_id", -1) >= 0]
    t_sent_ids = [m["sent_id"] for m in doc["vertexSet"][t_idx] if m.get("sent_id", -1) >= 0]
    if not h_sent_ids or not t_sent_ids:
        return float('inf')
    return min(abs(h - t) for h in h_sent_ids for t in t_sent_ids)


def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    """
    Insert entity markers (*) at the beginning and end of each entity mention.
    Returns tokenized sents, mapping from word indices to token positions, and sentence positions.
    """
    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
        new_map = {}
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        sent_end = len(sents)
        sent_pos.append((sent_start, sent_end))
        sent_start = sent_end
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)
    return sents, sent_map, sent_pos


def read_docred(file_in=None, tokenizer=None, max_seq_length=1024,
                curriculum_stage=0, stage1_max_dist=1, stage2_max_dist=4,
                data=None):
    """
    Load and preprocess a DocRED-style dataset.
    
    Args:
        file_in: path to JSON file (if data is None)
        tokenizer: HuggingFace tokenizer
        max_seq_length: maximum sequence length
        curriculum_stage: 0 = all pairs, 1 = easy (dist ≤1), 2 = medium (1<dist≤4), 3 = hard (dist>4)
        stage1_max_dist, stage2_max_dist: distance thresholds
        data: optional list of document dicts (if provided, file_in is ignored)

    Returns:
        List of feature dicts, each with keys: input_ids, entity_pos, labels, hts,
        sent_pos, sent_labels, distances, title
    """
    if data is None and file_in is None:
        raise ValueError("Either file_in or data must be provided")
    if data is None:
        with open(file_in, 'r') as fh:
            data = json.load(fh)

    features = []
    pos_samples = 0
    neg_samples = 0

    for sample in tqdm(data, desc="Processing documents"):
        entities = sample['vertexSet']
        # Collect all entity mention start and end positions (sentence, word index)
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0]))
                entity_end.append((sent_id, pos[1] - 1))

        # Add entity markers and get tokenized sentences
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # Build training triple dictionary: (h_idx, t_idx) -> list of {relation, evidence}
        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label.get('evidence', [])
                r = int(docred_rel2id[label['r']])
                key = (label['h'], label['t'])
                if key not in train_triple:
                    train_triple[key] = [{'relation': r, 'evidence': evidence}]
                else:
                    train_triple[key].append({'relation': r, 'evidence': evidence})

        # Entity positions (token indices) for each entity mention
        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end))

        # Build all possible entity pairs (h != t)
        num_entities = len(entities)
        total_pairs = num_entities * (num_entities - 1)
        all_hts = [None] * total_pairs
        all_labels = [None] * total_pairs
        all_sent_labels = [None] * total_pairs
        all_distances = [None] * total_pairs

        # Create mapping from (h,t) to index
        pair_to_idx = {}
        idx = 0
        for h in range(num_entities):
            for t in range(num_entities):
                if h == t:
                    continue
                pair_to_idx[(h, t)] = idx
                idx += 1

        # Fill positive pairs first
        for (h, t), mentions in train_triple.items():
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)
            for m in mentions:
                relation[m['relation']] = 1
                for evi in m['evidence']:
                    if evi < len(sent_evi):
                        sent_evi[evi] += 1
            idx = pair_to_idx[(h, t)]
            all_labels[idx] = relation
            all_hts[idx] = [h, t]
            all_sent_labels[idx] = sent_evi
            all_distances[idx] = get_entity_distance(sample, h, t)
            pos_samples += 1

        # Fill negative pairs (all remaining)
        for h in range(num_entities):
            for t in range(num_entities):
                if h == t:
                    continue
                if (h, t) in train_triple:
                    continue
                idx = pair_to_idx[(h, t)]
                relation = [1] + [0] * (len(docred_rel2id) - 1)   # NA relation at index 0
                sent_evi = [0] * len(sent_pos)
                all_labels[idx] = relation
                all_hts[idx] = [h, t]
                all_sent_labels[idx] = sent_evi
                all_distances[idx] = get_entity_distance(sample, h, t)
                neg_samples += 1

        # Sanity check
        assert None not in all_labels, f"Missing pairs in document {sample['title']}"

        # Apply curriculum stage filtering
        if curriculum_stage > 0:
            filtered_hts = []
            filtered_labels = []
            filtered_sent_labels = []
            filtered_distances = []
            for i, (h, t) in enumerate(all_hts):
                dist = all_distances[i]
                if (curriculum_stage == 1 and dist <= stage1_max_dist) or \
                   (curriculum_stage == 2 and stage1_max_dist < dist <= stage2_max_dist) or \
                   (curriculum_stage == 3 and dist > stage2_max_dist):
                    filtered_hts.append(all_hts[i])
                    filtered_labels.append(all_labels[i])
                    filtered_sent_labels.append(all_sent_labels[i])
                    filtered_distances.append(all_distances[i])
            all_hts = filtered_hts
            all_labels = filtered_labels
            all_sent_labels = filtered_sent_labels
            all_distances = filtered_distances

        # Skip document if no pairs remain after filtering
        if len(all_hts) == 0:
            continue

        # Truncate to max_seq_length - 2 for [CLS] and [SEP]
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Create a single feature for this document (document-level)
        feature = {
            'input_ids': input_ids,
            'entity_pos': entity_pos,
            'labels': all_labels,
            'hts': all_hts,
            'sent_pos': sent_pos,
            'sent_labels': all_sent_labels,
            'distances': all_distances,
            'title': sample['title']
        }
        features.append(feature)

    print(f"# of documents: {len(features)}")
    print(f"# of positive examples: {pos_samples}")
    print(f"# of negative examples: {neg_samples}")
    return features
