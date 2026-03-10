from config.env import load_project_env, login_wandb_from_env
load_project_env()
import pprint
import torch
import argparse
import os

from ppm.engine.distill import distill_engine
from ppm.models import NextEventPredictor
from ppm.models.config import get_model_config, get_model_params

from ppm.data_preparation.data_preparation import NUMERICAL_FEATURES, charge_loaders

from ppm.models.config import FreezeConfig

from ppm.models.nanoGPT import GPT, GPTConfig
from config.paths import get_paths  

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

#RANDOM_SEED = 42
#torch.manual_seed(RANDOM_SEED)


def parse_args():
    paths = get_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_path", type=str, default=str(paths.model_path("/app/persisted_models/best/")))
    parser.add_argument("--t_model_name", type=str, default="BPI12_qwen3-0.6b_run_ly8zeezl.pth")
    
    parser.add_argument("--dataset", type=str, default="BPI12")
    parser.add_argument("--lifecycle", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=True)
 
    #parser.add_argument("--persist_model", action="store_true", default=False)
    parser.add_argument("--append_run_info", action="store_true", default=False, help="Append run ID and val loss to saved model filename")
    parser.add_argument("--project_name", type=str, default="test_distillation")

    """ training config """
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512) # default=16 (before)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",  # current behavior
        choices = ["fp32", "tf32", "bf16", "fp16", "fp8", "fp4"],
        help="Numerics precision: fp32 (exact), tf32 (fast fp32 on Ampere), bf16, or fp16 (mixed)"
        )
    parser.add_argument("--compile", action="store_true", default=False)  #Cannot compile nanoGPT model, it is too dynamic
    parser.add_argument("--patience", type=int, default=1, help="for early stopping") # default=16 (before)
    parser.add_argument("--min_delta", type=float, default=0.05, help="minimum delta for the patience in the early stopping")
    parser.add_argument("--memory_safety_margin", type=float, default=None)
    parser.add_argument("--val_size", type=float, default=0.02, help="Size of the validation set relative to training set (e.g. 0.2 = 80/20 train/val split). 0 means the test set will be used instead (bad practice).")
    parser.add_argument("--val_split", type=str, default="classic", choices=["classic", "prefix"], help="Split strategy for both val: 'classic' (case-level) or 'prefix' (prefix-based)")
    
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")

    """ distillation config """
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_start", type=float, default=0.9)
    parser.add_argument("--alpha_end", type=float, default=0.1)

    """ features and tasks """
    # e.g.: python main --categorical_features a b
    parser.add_argument("--categorical_features", nargs="+", default=['activity', 'resource']) # default=None (before), now [] for easier debugging.
    parser.add_argument("--categorical_targets", nargs="+", default=['activity']) 
    parser.add_argument("--continuous_features", nargs="*", default=['accumulated_time', 'amount'])
    #parser.add_argument("--continuous_features", nargs="+", default=['accumulated_time',
    #                     'day_of_month',
    #                     'day_of_week',
    #                     'day_of_year',
    #                     'hour_of_day',
    #                     'min_of_hour',
    #                     'month_of_year',
    #                     'sec_of_min',
    #                     'secs_within_day',
    #                     'week_of_year']) # default=None (before), now [] for easier debugging.
    parser.add_argument("--continuous_targets", nargs="+", default=None)

    """ in layer config """
    parser.add_argument(
        "--strategy", type=str, default="concat", choices=["sum", "concat"]
    )

    """ model config """
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)  # GPT-2 has 12
    parser.add_argument("--n_heads", type=int, default=12)  
    parser.add_argument("--context_window", type=int, default=128)  
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
    
    ## check the arguments
    args = parser.parse_args()
    #_validate_args(parser, args)
    return args


def main(training_config: dict):

    train_log, train_loader, test_loader, val_loader, dataset_info = charge_loaders(training_config)
    
    teacher_path = os.path.join(training_config["t_path"], training_config["t_model_name"])
    print("teacher_path:", teacher_path)
    _safe_globals = getattr(torch.serialization, "safe_globals", None)
    ctx = _safe_globals([FreezeConfig]) if _safe_globals else open(os.devnull)
    with ctx:
        checkpoint = torch.load(teacher_path, map_location=training_config["device"], weights_only=False)
        print("Saved class:", checkpoint["model_class"])
        print("Saved module:", checkpoint["model_module"])
        print("Saved config:", checkpoint["model_config"])
        # Dynamically import the model class
        import importlib
        model_module = importlib.import_module(checkpoint["model_module"])
        model_class  = getattr(model_module, checkpoint["model_class"])

        # Recreate model from config
        checkpoint["model_config"]["student_model"] = None 
        teacher_model = model_class(**checkpoint["model_config"])

        # Load weights
        teacher_model.load_state_dict(checkpoint["net"])

        teacher_model.to(training_config["device"])
        teacher_model.eval()
    print("teacher loaded")

    
    # configure a small GPT2
    student_config = GPTConfig(
        vocab_size=5, #5 is placeholder. Gets overruled by Inlayer
        block_size=training_config["context_window"],   
        n_layer=training_config["n_layers"],        
        n_head=training_config["n_heads"],
        n_embd=training_config["hidden_size"],      
        )

    student_model = GPT(student_config)#.to(training_config["device"])
    
    student_model_config = get_model_config(train_log, training_config)
    # ensure the in-layer and out-layer width matches the student’s channel size
    student_model_config["embedding_size"] = student_model.config.n_embd
    student_model_config["backbone_hidden_size"] = student_model.config.n_embd
    # also ensure we actually use the student backbone
    student_model_config["student_model"] = student_model

    student_model = NextEventPredictor(**student_model_config).to(device=training_config["device"])
    
    #student_model = NextEventPredictor(**student_model_config).to(device=training_config["device"])
            
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    all_param, trainable_params = get_model_params(student_model)
    training_config.update(
        {
            "total_params": all_param,
            "trainable_params": trainable_params,
        }
    )

    use_wandb = training_config.pop("wandb")
    #persist_model = training_config.pop("persist_model")
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
        wandb.watch(student_model, log="all")
        wandb.log({f"dataset/{k}": v for k, v in dataset_info.items()}, step=0)
        
    use_val = training_config["val_size"] > 0

    print("=" * 80)
    print("Training")
    distill_engine(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader if use_val else test_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=training_config,
        use_wandb=use_wandb,
        #persist_model=persist_model,
        append_run_info=append_run_info,
        model_config=student_model_config,
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
        "lifecycle": True,
        "wandb": args.wandb,
        #"persist_model": args.persist_model,
        "append_run_info": args.append_run_info,
        # teacher
        "t_path": args.t_path,
        "t_model_name": args.t_model_name,
        # args to log
        "log": args.dataset,
        "device": args.device,
        # architecture
        "backbone": "student_model", #args.backbone,
        #"rnn_type": args.rnn_type,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "context_window": args.context_window,
        "time_positional_encoding": False, 
        "weight_tying": args.weight_tying,
        # hyperparameters
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "epochs": args.epochs,
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
        "val_split": args.val_split,
        "temperature": args.temperature,
        "alpha_start": args.alpha_start,
        "alpha_end": args.alpha_end,
        "num_workers": args.num_workers,
        "use_val": 0.0,
    }
    
    pprint.pprint(training_config)
    print("=" * 80)
    
    #training_config["lifecycle"] = True   #just for testing
    main(training_config)
