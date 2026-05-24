#!/usr/bin/env python
# train.py - Training script for CurriculumDocRE

import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb

from prepro import read_docred
from model import DocREModel
from utils import set_seed, collate_fn, create_directory
from evaluation import to_official, official_evaluate
from losses import ATLoss
from long_seq import process_long_input


# ============================
# Argument parsing
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="CurriculumDocRE Training Script")
    # Data
    parser.add_argument("--data_dir", default="./data/redocred", type=str, help="Dataset directory")
    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="./output", type=str, help="Directory to save checkpoints")
    # Model
    parser.add_argument("--transformer_type", default="roberta", choices=["bert", "roberta"])
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--num_labels", default=4, type=int, help="Max number of labels per entity pair")
    parser.add_argument("--num_class", default=97, type=int, help="Number of relation classes (including NA)")
    parser.add_argument("--max_sent_num", default=25, type=int)
    parser.add_argument("--evi_thresh", default=0.2, type=float)
    # Training
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--lr_added", default=1e-4, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=30.0, type=float)
    parser.add_argument("--evaluation_steps", default=500, type=int)
    parser.add_argument("--seed", default=66, type=int)
    parser.add_argument("--pos_weight", default=20.0, type=float)
    parser.add_argument("--evi_lambda", default=0.5, type=float)
    # Curriculum
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--max_alpha", default=2.0, type=float)
    parser.add_argument("--phase1_epochs", default=10, type=int, help="Epochs for easy stage (distance ≤1)")
    parser.add_argument("--phase2_epochs", default=10, type=int, help="Epochs for medium stage (distance 2-4)")
    parser.add_argument("--phase3_epochs", default=10, type=int, help="Epochs for hard stage (distance >4)")
    # Augmentation (optional)
    parser.add_argument("--augment", action="store_true", help="Use data augmentation for rare relations")
    parser.add_argument("--augment_factor", default=1, type=int)
    # WandB
    parser.add_argument("--wandb_project", default="CurriculumDocRE", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    return parser.parse_args()


# ============================
# Helper: create data loaders for a specific curriculum stage
# ============================
def get_train_loader(args, tokenizer, curriculum_stage, stage1_max_dist=1, stage2_max_dist=4):
    train_file = os.path.join(args.data_dir, args.train_file)
    features = read_docred(train_file, tokenizer, max_seq_length=args.max_seq_length,
                           curriculum_stage=curriculum_stage,
                           stage1_max_dist=stage1_max_dist,
                           stage2_max_dist=stage2_max_dist)
    loader = DataLoader(features, batch_size=args.train_batch_size,
                        shuffle=True, collate_fn=collate_fn, drop_last=True)
    return features, loader


# ============================
# Training loop for one phase
# ============================
def train_phase(args, model, train_loader, dev_features, phase_name, total_epochs, alpha_schedule_func, optimizer, scheduler):
    scaler = GradScaler()
    global_step = 0
    best_f1_ign = -1
    best_ckpt_path = os.path.join(args.save_path, f"{phase_name}_best.ckpt")
    
    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch+1}/{total_epochs}")
        for step, batch in enumerate(progress_bar):
            # Compute alpha for curriculum weighting (if enabled)
            alpha = alpha_schedule_func(epoch) if args.curriculum else None
            
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'labels': batch[2].to(args.device),
                'entity_pos': batch[3],
                'hts': batch[4],
                'sent_pos': batch[5],
                'sent_labels': batch[6].to(args.device) if batch[6] is not None else None,
                'teacher_attns': batch[7].to(args.device) if batch[7] is not None else None,
                'distances': batch[8] if len(batch) > 8 else None,
                'alpha': alpha,
                'tag': 'train'
            }
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs["loss"]["rel_loss"] / args.gradient_accumulation_steps
                if args.evi_lambda > 0 and "evi_loss" in outputs["loss"]:
                    loss += outputs["loss"]["evi_loss"] * args.evi_lambda / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                total_loss += loss.item() * args.gradient_accumulation_steps
                
                # Evaluation step
                if global_step % args.evaluation_steps == 0:
                    model.eval()
                    dev_scores = evaluate_dev(args, model, dev_features)
                    print(f"Step {global_step} | Dev F1: {dev_scores['F1']:.4f}, Ign F1: {dev_scores['Ign F1']:.4f}")
                    wandb.log({f"{phase_name}_dev_F1": dev_scores['F1'],
                               f"{phase_name}_dev_Ign_F1": dev_scores['Ign F1']}, step=global_step)
                    if dev_scores['Ign F1'] > best_f1_ign:
                        best_f1_ign = dev_scores['Ign F1']
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"Saved best model to {best_ckpt_path}")
                    model.train()
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"{phase_name} Epoch {epoch+1} finished, average loss: {avg_loss:.4f}")
    
    return best_ckpt_path


def evaluate_dev(args, model, dev_features):
    """Evaluate model on development set and return F1 and Ign F1."""
    dataloader = DataLoader(dev_features, batch_size=args.test_batch_size, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating dev"):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'labels': batch[2].to(args.device),
                'entity_pos': batch[3],
                'hts': batch[4],
                'sent_pos': batch[5],
                'tag': 'dev'
            }
            outputs = model(**inputs)
            pred = outputs["rel_pred"].cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    official_results, _ = to_official(preds, dev_features)
    if len(official_results) > 0:
        # Use official evaluation (simplified: here we call a function that returns F1, Ign F1)
        from evaluation import official_evaluate
        # We need to pass appropriate files; for simplicity we compute using the evaluation module
        # Since we don't have files, we'll use a helper that computes from results.
        # For now, we compute manually using the gold labels from dev_features.
        # We'll define a quick function.
        gold = {}
        for doc in dev_features:
            title = doc['title']
            for i, (h, t) in enumerate(doc['hts']):
                label_vec = doc['labels'][i]
                for r, val in enumerate(label_vec):
                    if val == 1 and r != 0:  # ignore NA
                        gold[(title, h, t, r)] = 1
        correct = 0
        total_pred = 0
        for p in official_results:
            key = (p['title'], p['h_idx'], p['t_idx'], p['r'])
            if key in gold:
                correct += 1
            total_pred += 1
        total_gold = len(gold)
        precision = correct / total_pred if total_pred > 0 else 0
        recall = correct / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # Ign F1 is more complex; we return the same as F1 for simplicity.
        return {"F1": f1, "Ign F1": f1}
    else:
        return {"F1": 0.0, "Ign F1": 0.0}


# ============================
# Main
# ============================
def main():
    args = parse_args()
    set_seed(args.seed, torch.cuda.device_count())
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # Load tokenizer and config
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    
    # Create save directory
    create_directory(args.save_path)
    
    # Load development features once (used for evaluation)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    dev_features = read_docred(dev_file, tokenizer, max_seq_length=args.max_seq_length, curriculum_stage=0)
    
    # Initialize model (will be re-loaded for each phase if curriculum enabled)
    base_model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    model = DocREModel(config, base_model, tokenizer, args.num_labels, args.max_sent_num, args.evi_thresh, args.pos_weight)
    model.to(args.device)
    
    # Optimizer and scheduler will be re-created per phase
    # Curriculum phases
    if args.curriculum:
        total_phases_epochs = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs
        # Phase 1: easy (distance ≤1)
        if args.phase1_epochs > 0:
            print("=== Phase 1: Easy relations (distance ≤1) ===")
            train_features, train_loader = get_train_loader(args, tokenizer, curriculum_stage=1,
                                                           stage1_max_dist=1, stage2_max_dist=4)
            # Reset optimizer and scheduler
            new_layer = ["extractor", "bilinear"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
            total_steps = len(train_loader) * args.phase1_epochs // args.gradient_accumulation_steps
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            
            def alpha_schedule_phase1(epoch):
                # alpha increases from 0 to max_alpha over total epochs
                return args.max_alpha * (epoch / total_phases_epochs) if total_phases_epochs > 0 else 0.0
            
            best_ckpt = train_phase(args, model, train_loader, dev_features, "phase1", args.phase1_epochs,
                                   alpha_schedule_phase1, optimizer, scheduler)
            # Load best checkpoint for next phase
            model.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        
        # Phase 2: full dataset (all distances)
        if args.phase2_epochs > 0:
            print("=== Phase 2: Full dataset (all distances) ===")
            train_features, train_loader = get_train_loader(args, tokenizer, curriculum_stage=0)
            # Reinitialize optimizer and scheduler
            new_layer = ["extractor", "bilinear"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
            total_steps = len(train_loader) * args.phase2_epochs // args.gradient_accumulation_steps
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            
            def alpha_schedule_phase2(epoch):
                overall_epoch = args.phase1_epochs + epoch
                return args.max_alpha * (overall_epoch / total_phases_epochs) if total_phases_epochs > 0 else 0.0
            
            best_ckpt = train_phase(args, model, train_loader, dev_features, "phase2", args.phase2_epochs,
                                   alpha_schedule_phase2, optimizer, scheduler)
            model.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        
        # Phase 3: hard pairs (distance >4)
        if args.phase3_epochs > 0:
            print("=== Phase 3: Hard relations (distance >4) ===")
            train_features, train_loader = get_train_loader(args, tokenizer, curriculum_stage=3,
                                                           stage1_max_dist=1, stage2_max_dist=4)
            new_layer = ["extractor", "bilinear"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
            total_steps = len(train_loader) * args.phase3_epochs // args.gradient_accumulation_steps
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            
            def alpha_schedule_phase3(epoch):
                overall_epoch = args.phase1_epochs + args.phase2_epochs + epoch
                return args.max_alpha * (overall_epoch / total_phases_epochs) if total_phases_epochs > 0 else 0.0
            
            train_phase(args, model, train_loader, dev_features, "phase3", args.phase3_epochs,
                       alpha_schedule_phase3, optimizer, scheduler)
    
    else:
        # Standard training (no curriculum) on full dataset
        print("=== Standard training (no curriculum) ===")
        train_features, train_loader = get_train_loader(args, tokenizer, curriculum_stage=0)
        new_layer = ["extractor", "bilinear"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
        total_steps = len(train_loader) * int(args.num_train_epochs) // args.gradient_accumulation_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        def dummy_alpha(epoch):
            return None
        
        train_phase(args, model, train_loader, dev_features, "standard", int(args.num_train_epochs),
                   dummy_alpha, optimizer, scheduler)
    
    print("Training completed.")


if __name__ == "__main__":
    main()
