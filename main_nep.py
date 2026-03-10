from config.env import load_project_env, login_wandb_from_env
load_project_env()
import pprint
import torch
import argparse

from ppm.engine.nep import train_engine
from ppm.models import NextEventPredictor
from ppm.models.config import get_model_config, get_model_params
from ppm.models.nanoGPT import GPT, GPTConfig

from ppm.data_preparation.data_preparation import NUMERICAL_FEATURES, charge_loaders

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BPI12")
    parser.add_argument("--lifecycle", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)

    #parser.add_argument("--persist_model", action="store_true", default=False)
    parser.add_argument("--append_run_info", action="store_true", default=False, help="Append run ID and val loss to saved model filename")
    parser.add_argument("--project_name", type=str, default="no_name_run")

    """ training config """
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64) # default=16 (before)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",  # current behavior
        choices = ["fp32", "tf32", "bf16", "fp8", "fp4"],
        help="Numerics precision: fp32 (exact), tf32 (fast fp32 on Ampere), bf16, or fp16 (mixed)"
        )
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=10, help="for early stopping") # default=16 (before)
    parser.add_argument("--min_delta", type=float, default=0.005, help="minimum delta for the patience in the early stopping")
    parser.add_argument("--memory_safety_margin", type=float, default=None)
    parser.add_argument("--val_size", type=float, default=0.1, help="Size of the validation set relative to training set (e.g. 0.2 = 80/20 train/val split). 0 means the test set will be used instead (bad practice).")
    parser.add_argument("--val_split", type=str, default="prefix", choices=["classic", "prefix"], help="Split strategy for both val: 'classic' (case-level) or 'prefix' (prefix-based)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")

    """ features and tasks """
    # e.g.: python main --categorical_features a b
    parser.add_argument("--categorical_features", nargs="+", default=['activity']) # default=None (before), now [] for easier debugging.
    parser.add_argument("--categorical_targets", nargs="+", default=['activity']) 
    parser.add_argument("--continuous_features", nargs="*", default=['accumulated_time', 'amount'])
    parser.add_argument("--continuous_targets", nargs="+", default=None)

    """ in layer config """
    parser.add_argument(
        "--strategy", type=str, default="sum", choices=["sum", "concat"]
    )

    """ model config """
    parser.add_argument(
        "--backbone",
        type=str,
        default="gpt2-small", # default=RNN (before).
        choices=["llama32-1b", "rnn", "gpt-oss-20b", "gpt2-small", "distilgpt2", "gpt2-tiny", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2-mini",
                 "qwen25-05b", "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
                 "nanogpt"],
    )
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument(
        "--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"]
    )

    """ if fine-tuning """
    parser.add_argument(
        "--fine_tuning",
        type=str,
        default="freeze",
        choices=["lora", "freeze", "none"],
        help="Fine-tuning mode: lora, freeze, or none (full backbone training).",
    )
    # if lora
    parser.add_argument("--r", type=int, default=2)   # default=None (before), now 2 for easier debugging.
    parser.add_argument("--lora_alpha", type=int, default=4)   # default=None (before), now 4 for easier debugging.
    # if freeze
    parser.add_argument(
        "--freeze_layers",
        nargs="+",
        type=int,
        default=None,
        help="List of layer indices to freeze. If None, all layers are frozen.",
    )
    # time positional encoding
    parser.add_argument(
        "--time_positional_encoding", type=str, default=None, choices=["additive", "rotary"] , \
            help="Type of time positional encoding to use for transformer models."
    )
    
    parser.add_argument(
    "--use_checkpoint",
    action="store_true",
    help="Enable activation checkpointing for the backbone (memory-saving, slower). "
         "Do NOT use on LRZ/Qwen3 unless you know it's safe.",
    )
    parser.add_argument(
        "--weight_tying",
        dest="weight_tying",
        action="store_true",
        help="Tie categorical output head weights to corresponding input embeddings when compatible.",
    )
    parser.add_argument(
        "--no-weight_tying",
        dest="weight_tying",
        action="store_false",
        help="Disable weight tying between input embeddings and output heads.",
    )
    parser.set_defaults(weight_tying=True)

    # check the arguments
    args = parser.parse_args()
    if args.fine_tuning == "none":
        args.fine_tuning = None
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # no positional encoding for rnn
    if args.backbone == "rnn" and args.time_positional_encoding:
        parser.error("Positional encoding is not applicable for RNN backbone.")
    if args.backbone == "pm-gpt2" and args.time_positional_encoding =="rotary":
        parser.error("Rotational time_positional encoding is not applicable for GPT2 backbone. Choose 'additive' or None.")
    if args.backbone not in ["llama32-1b", "qwen25-05b"] and args.fine_tuning == "lora":
        parser.error("LoRA fine-tuning is only implemented for LLaMA and Qwen backbones.")
    if args.backbone == "gpt-oss-20b" and args.precision == "fp16":
        parser.error("gpt-oss-20b should not have --precision fp16")
    if args.use_checkpoint and args.compile:
        parser.error("Do not use --compile together with --use_checkpoint, as it may cause issues with some backbones. Please choose one or the other.")


def main(training_config: dict):

    train_log, train_loader, test_loader, val_loader, dataset_info = charge_loaders(training_config)

    is_nanogpt = training_config["backbone"] == "nanogpt"
    if is_nanogpt:
        training_config["backbone"] = "student_model"

    model_config = get_model_config(train_log, training_config)
    model_config["use_checkpoint"] = args.use_checkpoint

    if is_nanogpt:
        gpt_config = GPTConfig(
            vocab_size=5,
            block_size=1024,
            n_layer=training_config["n_layers"],
            n_head=training_config["n_heads"],
            n_embd=training_config["hidden_size"],
        )
        nano = GPT(gpt_config)
        model_config["embedding_size"] = nano.config.n_embd
        model_config["backbone_hidden_size"] = nano.config.n_embd
        model_config["student_model"] = nano
    else:
        model_config["student_model"] = None

    model = NextEventPredictor(**model_config).to(device=training_config["device"])
            
    all_param, trainable_params = get_model_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    training_config.update(
        {
            "total_params": all_param,
            "trainable_params": trainable_params,
            "embedding_size": model_config["embedding_size"],
        }
    )

    use_wandb = training_config.pop("wandb")
    append_run_info = training_config.pop("append_run_info")
    if use_wandb and WANDB_AVAILABLE:
        login_wandb_from_env(wandb)
        if (
            "freeze_layers" in training_config
            and training_config["freeze_layers"] is not None
        ):
            training_config["freeze_layers"] = ",".join(
                [str(i) for i in training_config["freeze_layers"]]
            )
        wandb.init(project=training_config.pop("project_name"), config=training_config)
        wandb.watch(model, log="all")
        wandb.log({f"dataset/{k}": v for k, v in dataset_info.items()}, step=0)

    use_val = training_config["val_size"] > 0

    print("=" * 80)
    print("Training")
    train_engine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if use_val else test_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=training_config,
        use_wandb=use_wandb,
        append_run_info=append_run_info,
        model_config=model_config,
        dataset_info=dataset_info,
    )
    print("=" * 80)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()     
     
    training_config = {
        # args to pop before logging
        "project_name": args.project_name,
        "lifecycle": args.lifecycle,
        "wandb": args.wandb,
        #"persist_model": args.persist_model,
        "append_run_info": args.append_run_info,
        # args to log
        "log": args.dataset,
        "device": args.device,
        # architecture
        #"backbone": "gpt2-medium",  #DEBUG
        "backbone": args.backbone,
        "rnn_type": args.rnn_type,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "hidden_size": args.hidden_size,
        "time_positional_encoding": args.time_positional_encoding, 
        "weight_tying": args.weight_tying,
        # hyperparameters
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "epochs": args.epochs,
        # fine-tuning
        "fine_tuning": args.fine_tuning,
        "r": args.r,  # LoRA
        "lora_alpha": args.lora_alpha,  # LoRA
        "freeze_layers": args.freeze_layers,  # Freeze
        # features and tasks
        "categorical_features": args.categorical_features,
        "continuous_features": (
            [] if args.continuous_features is None
            else NUMERICAL_FEATURES if "all" in args.continuous_features
            else args.continuous_features
        ),
        #"continuous_features": (
        #    NUMERICAL_FEATURES
        #    if "all" in args.continuous_features
        #    else args.continuous_features
        #),
        "categorical_targets": args.categorical_targets,
        "continuous_targets": args.continuous_targets,
        "strategy": args.strategy,
        "precision": args.precision,
        "compile": args.compile,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "memory_safety_margin": args.memory_safety_margin,
        "val_size": args.val_size,
        #"test_split": args.test_split,
        "val_split": args.val_split,
        #"task": args.task,
        "num_workers": args.num_workers,
    }
    # if is_duplicate(training_config):
    #     print("Duplicate configuration. Skipping...")
    #     exit(0)

    pprint.pprint(training_config)
    print("=" * 80)
    main(training_config)
