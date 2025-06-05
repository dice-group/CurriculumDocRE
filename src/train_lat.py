import wandb  
"""
Enhanced training script that integrates curriculum-aware attention,
data quality, and data augmentation modules with fixes for class imbalance.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from torch.optim import AdamW
from tqdm import tqdm
import logging
import json
import numpy as np

import src.config as config
import src.utils as utils
from src.model import EnhancedCurriculumDocREModel
from src.data_quality import enhance_data_quality, filter_low_quality_instances
from src.data_augmentation import augment_data
from src.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def focal_loss(logits, targets, alpha=0.4, gamma=2):
    """
    Focal Loss for better handling of class imbalance.
    
    Args:
        logits: Raw model outputs
        targets: Ground truth labels
        alpha: Weighting factor for positive examples (increased from 0.25 to 0.4)
        gamma: Focusing parameter
        
    Returns:
        Computed focal loss
    """
    BCE_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    pt = torch.exp(-BCE_loss)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_t * (1 - pt) ** gamma * BCE_loss
    
    return focal_loss.mean()

def analyze_logits_distribution(relation_logits, labels=None, threshold=0.5):
    """Analyze the distribution of raw logits to diagnose issues."""
    logits = relation_logits.detach().cpu().numpy()
    
    stats = {
        "min": float(np.min(logits)),
        "max": float(np.max(logits)),
        "mean": float(np.mean(logits)),
        "std": float(np.std(logits)),
    }

    if stats["max"] < 0:
        logger.warning("WARNING: All logits are negative, model will predict all zeros")
 
    if stats["std"] < 0.1:
        logger.warning("WARNING: Low variance in logits, model may not have learned meaningful patterns")
  
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]  
    pos_counts = {}
    
    for thresh in thresholds:
        pos_preds = (probs > thresh).sum()
        pos_counts[str(thresh)] = int(pos_preds)
        
        if pos_preds == 0:
            logger.warning(f"WARNING: No positive predictions at threshold {thresh}")
 
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        true_pos_count = int((labels_np == 1).sum())
        stats["true_positives"] = true_pos_count
        
        if true_pos_count > 0:
            for thresh in thresholds:
                preds = (probs > thresh).astype(np.float32)
                tp = ((preds == 1) & (labels_np == 1)).sum()
                stats[f"true_positives_at_{thresh}"] = int(tp)
    
    stats["positive_predictions"] = pos_counts
    return stats

def train_one_stage(model, dataloader, optimizer, cls_criterion, conf_criterion, device, stage, global_step=0, previous_best_threshold=0.05):
    """
    Train model for one curriculum stage with enhanced features and diagnostics.
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_conf_loss = 0
    step = global_step
    loss_history = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Stage {stage}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).float() 
        confidence = batch["confidence"].to(device).float()  

        relation_logits, confidence_logit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            current_curriculum_stage=stage
        )

        loss_cls = focal_loss(relation_logits, labels, alpha=0.4, gamma=2)
        loss_conf = conf_criterion(confidence_logit.view(-1), confidence.view(-1))
 
        conf_weight = 0.3 + (stage - 1) * 0.2  
        loss = loss_cls + conf_weight * loss_conf

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_conf_loss += loss_conf.item()
        loss_history.append(loss.item())
        step += 1
 
        if step % 2000 == 0 or (step < 2000 and step % 50 == 0):
            early_checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"model_step_{step}.pt")
            torch.save(model.state_dict(), early_checkpoint_path)
            logger.info(f"Saved early checkpoint to {early_checkpoint_path}")
    
            with torch.no_grad():
                logits_stats = analyze_logits_distribution(relation_logits, labels)
                logger.info(f"Step {step} logits stats: min={logits_stats['min']:.4f}, max={logits_stats['max']:.4f}, "
                           f"mean={logits_stats['mean']:.4f}, std={logits_stats['std']:.4f}")
  
                for thresh, count in logits_stats["positive_predictions"].items():
                    logger.info(f"Step {step}: {count} positive predictions at threshold {thresh}")

            with open(os.path.join(config.OUTPUT_DIR, "loss_history.json"), "w") as f:
                json.dump(loss_history, f)

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_conf_loss = total_conf_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_conf_loss, step, previous_best_threshold
import numpy as np
from tqdm import tqdm
import torch

def evaluate(model, dataloader, device, stage, previous_best_threshold=0.05):
    """
    Evaluate model with enhanced metrics and diagnostics for class imbalance.
    Uses dynamic thresholds based on output probability distribution.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Stage {stage}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            relation_logits, confidence_logit = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                current_curriculum_stage=stage
            )
            
            all_logits.append(relation_logits.cpu())

            prob_values = torch.sigmoid(relation_logits).view(-1).cpu().numpy()
            thresholds = list(np.unique(np.quantile(prob_values, [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]))[::-1])

            threshold_metrics = {}
            closest_threshold = min(thresholds, key=lambda t: abs(t - previous_best_threshold))

            for threshold in thresholds:
                preds = (torch.sigmoid(relation_logits) > threshold).float()

                if threshold == closest_threshold:  
                    correct += ((preds == labels).sum().item())
                    total += labels.numel()
                    all_preds.append(preds.cpu())

                batch_preds = preds.cpu()
                batch_labels = labels.cpu()

                true_positives = ((batch_preds == 1) & (batch_labels == 1)).sum().item()
                predicted_positives = (batch_preds == 1).sum().item()
                actual_positives = (batch_labels == 1).sum().item()

                threshold_metrics[str(threshold)] = {
                    "true_positives": true_positives,
                    "predicted_positives": predicted_positives,
                    "actual_positives": actual_positives
                }

            all_labels.append(labels.cpu())
            all_confidences.append(torch.sigmoid(confidence_logit).cpu())

            if len(all_logits) == 1:
                logits_stats = analyze_logits_distribution(relation_logits, labels)
                logger.info(f"Evaluation logits stats: min={logits_stats['min']:.4f}, max={logits_stats['max']:.4f}, "
                            f"mean={logits_stats['mean']:.4f}, std={logits_stats['std']:.4f}")

                for thresh, count in logits_stats["positive_predictions"].items():
                    logger.info(f"Evaluation: {count} positive predictions at threshold {thresh}")

                for thresh, metrics in threshold_metrics.items():
                    tp = metrics["true_positives"]
                    pp = metrics["predicted_positives"]
                    ap = metrics["actual_positives"]

                    precision = tp / pp if pp > 0 else 0
                    recall = tp / ap if ap > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    logger.info(f"Threshold {thresh}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    accuracy = correct / total if total > 0 else 0

    if len(all_preds) == 0:
        logger.warning("No predictions collected during evaluation — skipping metric computation.")
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "f2": 0,
            "confidence_accuracy": 0,
            "best_f1": 0,
            "best_threshold": previous_best_threshold,
            "best_f2": 0,
            "best_f2_threshold": previous_best_threshold,
            "threshold_results": {}
        }, previous_best_threshold

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_confidences = torch.cat(all_confidences, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    true_positives = ((all_preds == 1) & (all_labels == 1)).sum().item()
    predicted_positives = (all_preds == 1).sum().item()
    actual_positives = (all_labels == 1).sum().item()

    logger.info(f"Evaluation summary:")
    logger.info(f"  Total predictions: {all_preds.numel()}")
    logger.info(f"  Positive predictions: {predicted_positives} ({predicted_positives/all_preds.numel()*100:.2f}%)")
    logger.info(f"  Actual positives: {actual_positives} ({actual_positives/all_preds.numel()*100:.2f}%)")
    logger.info(f"  True positives: {true_positives}")

    if predicted_positives == 0:
        logger.warning("MODEL IS PREDICTING ALL NEGATIVES - ADJUST THRESHOLD OR CHECK TRAINING")

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    beta = 2
    f2 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0

    confidence_weighted_preds = all_preds * all_confidences
    confidence_accuracy = ((confidence_weighted_preds > 0.5) == (all_labels == 1)).float().mean().item()

   
    all_probs = torch.sigmoid(all_logits).view(-1).cpu().numpy()
    thresholds = list(np.unique(np.quantile(all_probs, [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]))[::-1])

    best_f1 = 0
    best_f2 = 0
    best_threshold = previous_best_threshold
    best_f2_threshold = previous_best_threshold
    threshold_results = {}

    for threshold in thresholds:
        threshold_preds = (torch.sigmoid(all_logits) > threshold).float()
        threshold_tp = ((threshold_preds == 1) & (all_labels == 1)).sum().item()
        threshold_pp = (threshold_preds == 1).sum().item()

        threshold_precision = threshold_tp / threshold_pp if threshold_pp > 0 else 0
        threshold_recall = threshold_tp / actual_positives if actual_positives > 0 else 0
        threshold_f1 = 2 * threshold_precision * threshold_recall / (threshold_precision + threshold_recall) if (threshold_precision + threshold_recall) > 0 else 0
        threshold_f2 = (1 + beta**2) * threshold_precision * threshold_recall / ((beta**2 * threshold_precision) + threshold_recall) if (threshold_precision + threshold_recall) > 0 else 0

        threshold_results[str(threshold)] = {
            "precision": threshold_precision,
            "recall": threshold_recall,
            "f1": threshold_f1,
            "f2": threshold_f2,
            "positive_predictions": threshold_pp
        }

        logger.info(f"Threshold {threshold}: F1={threshold_f1:.4f}, F2={threshold_f2:.4f}, Precision={threshold_precision:.4f}, Recall={threshold_recall:.4f}, Positives={threshold_pp}")

        if threshold_f1 > best_f1:
            best_f1 = threshold_f1
            best_threshold = threshold

        if threshold_f2 > best_f2:
            best_f2 = threshold_f2
            best_f2_threshold = threshold

    logger.info(f"Best F1 threshold: {best_threshold} with F1={best_f1:.4f}")
    logger.info(f"Best F2 threshold: {best_f2_threshold} with F2={best_f2:.4f}")

    if best_threshold != previous_best_threshold:
        logger.info(f"Threshold changed from {previous_best_threshold} to {best_threshold}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "confidence_accuracy": confidence_accuracy,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_f2": best_f2,
        "best_f2_threshold": best_f2_threshold,
        "threshold_results": threshold_results
    }

    return metrics, best_threshold

import json

def safe_config(config_module):
    safe = {}
    for k, v in config_module.__dict__.items():
        try:
            json.dumps(v)  
            safe[k] = v
        except (TypeError, OverflowError):
            continue  
    return safe

def prepare_data_pipeline_full():
    """
    Prepare the enhanced data pipeline with quality checks and augmentation.
    """
    logger.info("Preparing enhanced data pipeline...")

    relation2id = utils.build_relation2id(config.DOCRED_REL_INFO_FILE)

    train_data = utils.load_json(config.DOCRED_TRAIN_FILE)

    logger.info("Enhancing data quality...")
    enhanced_data, uncertain_instances = enhance_data_quality(train_data, relation2id)
 
    logger.info("Filtering low quality instances...")
    filtered_data = filter_low_quality_instances(enhanced_data, confidence_threshold=0.3)
 
    logger.info("Augmenting data...")
    augmented_data = augment_data(filtered_data, augmentation_factor=2)

    utils.create_directories()
    
    processed_file = os.path.join(config.OUTPUT_DIR, "train_processed.json")
    with open(processed_file, "w") as f:
        json.dump(augmented_data, f)
    
    logger.info(f"Processed data saved to {processed_file}")
    logger.info(f"Original instances: {len(train_data)}")
    logger.info(f"Processed instances: {len(augmented_data)}")
    
    return processed_file, relation2id
def main():
    """
    Main training function with enhanced curriculum learning and diagnostics.
    """
    utils.set_seed(config.SEED)

    wandb.init(
        project="docre-curriculum",
        name="curriculum_run",
        config=safe_config(config)
        )

    utils.create_directories()

    processed_file, relation2id = prepare_data_pipeline_full()
    config.DOCRED_TRAIN_FILE = processed_file

    num_labels = len(relation2id)
    logger.info(f"Loaded {num_labels} relation types (including NA).")

    tokenizer = BertTokenizerFast.from_pretrained(config.BASE_MODEL_NAME)
    model = EnhancedCurriculumDocREModel(
        base_model_name=config.BASE_MODEL_NAME,
        num_relations=num_labels
    )

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Multiple GPUs detected: {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
    model.to(device)

    wandb.watch(model, log="all")  

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)

    pos_weight = torch.ones([num_labels], device=device) * 15
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    conf_criterion = nn.MSELoss()

    all_metrics = {}
    global_step = 0
    best_threshold = 0.05

    for stage in [1, 2, 3]:
        logger.info(f"\n--- Stage {stage} Training ---")
        train_loader, dev_loader = get_dataloaders(tokenizer, stage, config.BATCH_SIZE, relation2id)

        if not train_loader:
            logger.warning(f"No data found for Stage {stage}. Skipping.")
            continue

        stage_metrics = []
        for epoch in range(config.NUM_EPOCHS):
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

            avg_loss, avg_cls_loss, avg_conf_loss, global_step, best_threshold = train_one_stage(
                model, train_loader, optimizer, cls_criterion, conf_criterion, device, stage, global_step, best_threshold
            )

            metrics, best_threshold = evaluate(model, dev_loader, device, stage, best_threshold)

            logger.info(f"Stage {stage} - Epoch {epoch+1}:")
            logger.info(f"  Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Conf: {avg_conf_loss:.4f})")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.info(f"  F2 Score (recall-weighted): {metrics['f2']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  Best F1 (threshold={metrics['best_threshold']}): {metrics['best_f1']:.4f}")
            logger.info(f"  Best F2 (threshold={metrics['best_f2_threshold']}): {metrics['best_f2']:.4f}")

            wandb.log({
                "stage": stage,
                "epoch": epoch + 1,
                "total_loss": avg_loss,
                "classification_loss": avg_cls_loss,
                "confidence_loss": avg_conf_loss,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"],
                "f2_score": metrics["f2"],
                "best_f1": metrics["best_f1"],
                "best_threshold": metrics["best_threshold"],
                "best_f2": metrics["best_f2"],
                "best_f2_threshold": metrics["best_f2_threshold"]
            })

            stage_metrics.append(metrics)

            epoch_checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"model_stage{stage}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), epoch_checkpoint_path)
            logger.info(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

        all_metrics[f"stage_{stage}"] = stage_metrics

    logger.info(f"Saving final model to {config.MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    metrics_file = os.path.join(config.RESULTS_SAVE_DIR, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Training metrics saved to {metrics_file}")
    logger.info("Training complete.")

    wandb.finish()

if __name__ == "__main__":
    main()
