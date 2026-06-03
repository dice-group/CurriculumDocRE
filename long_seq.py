import torch

def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    """
    Simplified version: assumes input length ≤ max_seq_length (no chunking).
    Returns sequence_output and attention (averaged over last 3 layers).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True
    )
    
    # Sequence output: average of last 3 hidden states
    if outputs.hidden_states is not None:
        hidden_states = outputs.hidden_states[-3:] if len(outputs.hidden_states) >= 3 else outputs.hidden_states
        seq_out = torch.stack(hidden_states, dim=1).mean(dim=1)
    else:
        seq_out = outputs.last_hidden_state
    
    # Attention: average of last 3 attention matrices
    if outputs.attentions is not None and len(outputs.attentions) > 0:
        attns = outputs.attentions[-3:] if len(outputs.attentions) >= 3 else outputs.attentions
        attn = torch.stack(attns, dim=1).mean(dim=1)
    else:
        # Dummy identity attention (fallback)
        bsz, seq_len = input_ids.size()
        attn = torch.eye(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
    
    return seq_out, attn
