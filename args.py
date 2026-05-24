# args.py - Command line argument definitions for CurriculumDocRE

import argparse


def add_args(parser):
    """
    Add all arguments to the given parser.
    Returns the same parser for convenience.
    """
    # -------------------- Basic runtime flags --------------------
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation")
    parser.add_argument("--test_only", action="store_true", help="Run evaluation on test set only")
    parser.add_argument("--evaluate", action="store_true", help="Alias for --do_eval")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation for rare relations")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--save_attn", action="store_true", help="Save attention weights (teacher mode)")

    # -------------------- Data paths --------------------
    parser.add_argument("--data_dir", default="./data/redocred", type=str, help="Dataset directory")
    parser.add_argument("--train_file", default="train.json", type=str, help="Training file name")
    parser.add_argument("--dev_file", default="dev.json", type=str, help="Development file name")
    parser.add_argument("--test_file", default="test.json", type=str, help="Test file name")
    parser.add_argument("--meta_dir", default="meta", type=str, help="Directory with rel2id.json etc.")
    parser.add_argument("--save_path", default="./output", type=str, help="Directory to save checkpoints/results")
    parser.add_argument("--load_path", default="", type=str, help="Path to load checkpoint for evaluation or resuming")
    parser.add_argument("--teacher_sig_path", default="", type=str, help="Path to teacher attention signals")

    # -------------------- Model architecture --------------------
    parser.add_argument("--transformer_type", default="roberta", choices=["bert", "roberta"],
                        help="Type of pre-trained language model")
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str,
                        help="Pretrained model name or local path")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name")
    parser.add_argument("--max_seq_length", default=1024, type=int, help="Maximum input tokens")
    parser.add_argument("--num_labels", default=4, type=int, help="Maximum number of relations per entity pair (ATLoss)")
    parser.add_argument("--num_class", default=97, type=int, help="Total number of relation types (including NA)")
    parser.add_argument("--max_sent_num", default=25, type=int, help="Maximum number of sentences per document")
    parser.add_argument("--evi_thresh", default=0.2, type=float, help="Threshold for evidence prediction")

    # -------------------- Training hyperparameters --------------------
    parser.add_argument("--train_batch_size", default=4, type=int, help="Training batch size per GPU")
    parser.add_argument("--test_batch_size", default=8, type=int, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="Learning rate for transformer")
    parser.add_argument("--lr_added", default=1e-4, type=float, help="Learning rate for newly added layers")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Gradient clipping norm")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup proportion")
    parser.add_argument("--num_train_epochs", default=30.0, type=float, help="Total training epochs")
    parser.add_argument("--evaluation_steps", default=500, type=int, help="Evaluate every N steps")
    parser.add_argument("--seed", default=66, type=int, help="Random seed")
    parser.add_argument("--pos_weight", default=20.0, type=float, help="Positive class weight for ATLoss")
    parser.add_argument("--evi_lambda", default=0.5, type=float, help="Weight for evidence loss")
    parser.add_argument("--attn_lambda", default=1.0, type=float, help="Weight for attention distillation loss")

    # -------------------- Curriculum learning --------------------
    parser.add_argument("--max_alpha", default=2.0, type=float, help="Final exponent for distance weighting")
    parser.add_argument("--phase1_epochs", default=10, type=int, help="Epochs for easy stage (distance ≤1)")
    parser.add_argument("--phase2_epochs", default=10, type=int, help="Epochs for medium stage (distance 2‑4)")
    parser.add_argument("--phase3_epochs", default=10, type=int, help="Epochs for hard stage (distance >4)")
    parser.add_argument("--stage1_max_dist", default=1, type=int, help="Max distance for stage 1")
    parser.add_argument("--stage2_max_dist", default=4, type=int, help="Max distance for stage 2 (exclusive lower bound)")

    # -------------------- Augmentation --------------------
    parser.add_argument("--augment_factor", default=1, type=int, help="Number of augmented copies per positive document")
    parser.add_argument("--entity_sub_prob", default=0.3, type=float, help="Probability of entity substitution")
    parser.add_argument("--evidence_mask_prob", default=0.2, type=float, help="Probability of evidence masking")
    parser.add_argument("--relation_transfer_prob", default=0.2, type=float, help="Probability of cross‑document transfer")

    # -------------------- Evaluation / inference --------------------
    parser.add_argument("--eval_mode", default="single", choices=["single", "fusion"],
                        help="Single-pass or inference‑stage fusion")
    parser.add_argument("--pred_file", default="results.json", type=str, help="Output prediction file name")

    # -------------------- WandB logging --------------------
    parser.add_argument("--wandb_project", default="CurriculumDocRE", type=str)
    parser.add_argument("--wandb_name", default=None, type=str, help="Run name for WandB")

    return parser


def get_args():
    """Parse command line arguments using the definitions above."""
    parser = argparse.ArgumentParser(description="CurriculumDocRE: Curriculum Learning for DocRE")
    parser = add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Quick test
    args = get_args()
    print(args)
