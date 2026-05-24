# model.py - DocREModel with curriculum learning support

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input


class ATLoss(nn.Module):
    """
    Adaptive Threshold Loss (ATLoss) for multi-label classification.
    Supports per-pair loss reduction and positive class weighting.
    """
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, labels, reduction='mean'):
        # TH label (no-relation class, index 0)
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        loss1 = loss1 * self.pos_weight

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        loss = loss1 + loss2
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):
        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0


class DocREModel(nn.Module):
    """
    Document-level relation extraction model with curriculum learning.
    Based on DREEAM/ATLOP, with added support for distance-based loss weighting.
    """
    def __init__(self, config, model, tokenizer,
                 num_labels=4, max_sent_num=25, evi_thresh=0.2, pos_weight=1.0):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh

        self.loss_fnt = ATLoss(pos_weight=pos_weight)
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        # Head and tail extractors
        self.head_extractor = nn.Linear(2 * self.hidden_size, 768)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, 768)
        self.bilinear = nn.Linear(768 * 64, config.num_labels)  # block_size=64

    def encode(self, input_ids, attention_mask):
        """Encode document and return token embeddings and attention."""
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask,
                                                         start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):
        """
        Obtain head, relation, and tail representations for each entity pair.
        Also returns attention weights for evidence extraction.
        """
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        batch_rel = []   # number of pairs per document in batch
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            # Entity-level pooling (logsumexp over mentions)
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)   # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)   # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)                 # average over heads
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)

            # Local context pooling
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            batch_rel.append(ht_i.size(0))

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, batch_rel

    def forward_rel(self, hs, ts, rs):
        """Bilinear classifier for relation prediction."""
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        # Group bilinear (block_size=64)
        b1 = hs.view(-1, 768 // 64, 64)
        b2 = ts.view(-1, 768 // 64, 64)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, 768 * 64)
        logits = self.bilinear(bl)
        return logits

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        """Evidence sentence prediction from attention weights."""
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)):
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]
            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1)
            rel_sent_attn.append(sum_attn)
        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                sent_pos=None,
                sent_labels=None,
                teacher_attns=None,
                distances=None,
                alpha=None,
                tag="train"):
        """
        Forward pass.
        Args:
            input_ids, attention_mask: tokenized document.
            labels: relation labels (one-hot per pair, concatenated across documents in batch).
            entity_pos: list of entity mention token positions per document.
            hts: list of head-tail indices per document.
            sent_pos: list of sentence token positions per document.
            sent_labels: gold evidence labels (per pair, per sentence).
            teacher_attns: attention from teacher (for distillation).
            distances: list of per-pair sentence distances (for curriculum weighting).
            alpha: exponent for curriculum weighting (if None, no weighting).
            tag: 'train', 'dev', 'test', 'infer'.
        Returns:
            dict containing predictions and losses.
        """
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts, offset)
        logits = self.forward_rel(hs, ts, rs)

        # Relation predictions (for inference)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        if tag in ["test", "dev"]:
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]

        # Evidence prediction (if sent_pos is provided)
        if sent_pos is not None:
            # Compute token-level attention (average over heads) for evidence extraction
            doc_attn = attention.mean(dim=1) if attention is not None else None
            if doc_attn is not None:
                s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
                output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        # Loss computation (training only)
        output["loss"] = {}
        if tag == "train" and labels is not None:
            # Relation loss with optional curriculum weighting
            if distances is not None and alpha is not None and self.training:
                # Flatten distances (list of lists -> 1D tensor)
                flat_dist = []
                for d in distances:
                    if isinstance(d, (list, tuple)):
                        flat_dist.extend(d)
                    else:
                        flat_dist.append(d)
                flat_dist = torch.tensor(flat_dist, device=logits.device, dtype=torch.float)
                norm_dist = torch.clamp(flat_dist / 20.0, max=1.0)
                weight = norm_dist ** alpha
                per_pair_loss = self.loss_fnt(logits, labels.float(), reduction='none')
                weighted_loss = (per_pair_loss * weight).mean()
                output["loss"]["rel_loss"] = weighted_loss
            else:
                rel_loss = self.loss_fnt(logits.float(), labels.float())
                output["loss"]["rel_loss"] = rel_loss

            # Evidence loss (if gold evidence labels available)
            if sent_labels is not None and sent_pos is not None and doc_attn is not None:
                s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
                # Only consider positive relation pairs (where label not NA)
                idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1)
                if len(idx_used) > 0:
                    s_attn = s_attn[idx_used]
                    sent_labels = sent_labels[idx_used]
                    norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                    norm_s_labels[norm_s_labels == 0] = 1e-30
                    s_attn[s_attn == 0] = 1e-30
                    evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                    output["loss"]["evi_loss"] = evi_loss

            # Teacher attention distillation loss (optional)
            if teacher_attns is not None:
                doc_attn = attention.mean(dim=1) if attention is not None else None
                if doc_attn is not None:
                    doc_attn[doc_attn == 0] = 1e-30
                    teacher_attns[teacher_attns == 0] = 1e-30
                    attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                    output["loss"]["attn_loss"] = attn_loss

        return output
