# utils.py - Utility functions for CurriculumDocRE

import os
import random
import numpy as np
import torch


def create_directory(d):
    """Create directory if it does not exist."""
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d


def set_seed(seed, n_gpu=1):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Supports both document-level features (multiple pairs per document)
    and flattened pair-level features (one pair per item).
    """
    # Determine max sequence length and max number of sentences
    max_len = max([len(f["input_ids"]) for f in batch])
    max_sent = max([len(f["sent_pos"]) for f in batch]) if "sent_pos" in batch[0] else 0

    # Pad input_ids and attention_mask
    input_ids = []
    attention_mask = []
    for f in batch:
        pad_len = max_len - len(f["input_ids"])
        ids = torch.cat([f["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        mask = torch.cat([torch.ones(len(f["input_ids"]), dtype=torch.float),
                          torch.zeros(pad_len, dtype=torch.float)])
        input_ids.append(ids)
        attention_mask.append(mask)

    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)

    # Labels: can be stored as 'labels' (document-level) or 'label' (pair-level)
    if "labels" in batch[0]:
        labels = [torch.tensor(f["labels"], dtype=torch.float32) for f in batch]
        labels = torch.cat(labels, dim=0)
    elif "label" in batch[0]:
        labels = torch.stack([f["label"] for f in batch])
    else:
        labels = None

    # Entity positions and head-tail pairs
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]

    # Sentence positions and labels (evidence)
    sent_pos = [f["sent_pos"] for f in batch] if "sent_pos" in batch[0] else []
    if "sent_labels" in batch[0]:
        sent_labels = [torch.tensor(f["sent_labels"], dtype=torch.float32) for f in batch]
        # Pad sentence dimension to max_sent
        sent_labels_padded = []
        for sl in sent_labels:
            if sl.size(0) < max_sent:
                pad = torch.zeros(max_sent - sl.size(0), dtype=sl.dtype)
                sl = torch.cat([sl, pad])
            sent_labels_padded.append(sl)
        sent_labels_tensor = torch.stack(sent_labels_padded, dim=0)
    elif "sent_label" in batch[0]:
        sent_labels_tensor = torch.stack([f["sent_label"] for f in batch])
        if sent_labels_tensor.size(1) < max_sent:
            pad = torch.zeros(sent_labels_tensor.size(0), max_sent - sent_labels_tensor.size(1), dtype=sent_labels_tensor.dtype)
            sent_labels_tensor = torch.cat([sent_labels_tensor, pad], dim=1)
    else:
        sent_labels_tensor = None

    # Distances (for curriculum weighting) – list of lists (document-level) or 1D tensor (pair-level)
    if "distances" in batch[0]:
        distances = [f["distances"] for f in batch]
    else:
        distances = None

    # Teacher attentions (optional)
    attns = [f["attns"] for f in batch] if "attns" in batch[0] else []
    if attns:
        attns_padded = []
        for attn in attns:
            pad_len = max_len - attn.shape[1]
            if pad_len > 0:
                pad = np.zeros((attn.shape[0], pad_len))
                attn_padded = np.concatenate([attn, pad], axis=1)
            else:
                attn_padded = attn
            attns_padded.append(attn_padded)
        attns_tensor = torch.from_numpy(np.concatenate(attns_padded, axis=0))
    else:
        attns_tensor = None

    return (input_ids, attention_mask, labels, entity_pos, hts, sent_pos, sent_labels_tensor, attns_tensor, distances)
