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

    # Labels: concatenate (different number of pairs per doc)
    labels = [torch.tensor(f["labels"], dtype=torch.float32) for f in batch]
    labels = torch.cat(labels, dim=0)

    # Entity positions and hts remain as lists of lists
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    sent_pos = [f["sent_pos"] for f in batch] if "sent_pos" in batch[0] else []

    # Evidence labels (sent_labels): pad sentence dimension and concatenate
    sent_labels_tensor = None
    if "sent_labels" in batch[0]:
        sent_labels_list = []
        for f in batch:
            sl = f["sent_labels"]
            sl_tensor = sl if isinstance(sl, torch.Tensor) else torch.tensor(sl, dtype=torch.float32)
            # sl_tensor shape: (num_pairs, num_sentences)
            if sl_tensor.size(1) < max_sent:
                pad = torch.zeros(sl_tensor.size(0), max_sent - sl_tensor.size(1), dtype=torch.float32)
                sl_tensor = torch.cat([sl_tensor, pad], dim=1)
            sent_labels_list.append(sl_tensor)
        sent_labels_tensor = torch.cat(sent_labels_list, dim=0)

    # Distances: list of lists (keep as is, will be flattened in model)
    distances = [f["distances"] for f in batch] if "distances" in batch[0] else []

    # Teacher attentions (if present)
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
        attns = torch.from_numpy(np.concatenate(attns_padded, axis=0))
    else:
        attns = None

    return (input_ids, attention_mask, labels, entity_pos, hts,
            sent_pos, sent_labels_tensor, attns, distances)
