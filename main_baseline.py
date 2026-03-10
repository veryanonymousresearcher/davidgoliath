from config.env import load_project_env, login_wandb_from_env
load_project_env()

import argparse
import random
from collections import Counter, defaultdict

import torch
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score

from ppm.data_preparation.data_preparation import charge_loaders
from ppm.wandb_utils import BASELINE_PROJECT

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline model for next event prediction using transition frequencies")
    parser.add_argument("--dataset", type=str, default="BPI12")
    parser.add_argument("--lifecycle", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default=BASELINE_PROJECT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_attention_mask(x_cat, y_cat):
    attention_mask = (x_cat[..., 0] != 0).long()
    attention_mask = attention_mask * (y_cat[..., 0] != -1).long()
    return attention_mask


def build_transition_counts(train_loader) -> dict[int, Counter]:
    transition_counts: dict[int, Counter] = defaultdict(Counter)
    
    for x_cat, x_num, y_cat, y_num in train_loader:
        attention_mask = compute_attention_mask(x_cat, y_cat)
        mask = attention_mask.bool()
        
        x = x_cat[..., 0]
        y = y_cat[..., 0]
        
        x_masked = x[mask]
        y_masked = y[mask]
        
        for i in range(x_masked.shape[0]):
            transition_counts[x_masked[i].item()][y_masked[i].item()] += 1
    
    return dict(transition_counts)


def build_predictions(transition_counts: dict[int, Counter], seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    predictions = {}

    for activity, counts in transition_counts.items():
        max_count = max(counts.values())
        candidates = [a for a, c in counts.items() if c == max_count]
        predictions[activity] = rng.choice(candidates)
    
    return predictions


def evaluate(test_loader, transition_counts: dict[int, Counter], predictions: dict[int, int]) -> tuple[float, float, float, int, int]:
    all_preds = []
    all_targets = []
    all_curr = []  # current activity for each evaluated position

    for x_cat, x_num, y_cat, y_num in test_loader:
        attention_mask = compute_attention_mask(x_cat, y_cat)
        mask = attention_mask.bool().view(-1)

        x = x_cat[..., 0].view(-1)
        y = y_cat[..., 0].view(-1)

        pred_tensor = torch.tensor([predictions.get(xi.item(), -1) for xi in x], dtype=y.dtype)

        all_preds.extend(pred_tensor[mask].tolist())
        all_targets.extend(y[mask].tolist())
        all_curr.extend(x[mask].tolist())

    total = len(all_targets)
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    accuracy = correct / total if total > 0 else 0.0
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0.0)

    # --- AUROC (macro) using transition-frequency probabilities as scores ---
    auroc_macro = float("nan")
    if total > 0:
        labels = sorted(set(all_targets))
        k = len(labels)

        if k >= 2:
            label_to_col = {lab: i for i, lab in enumerate(labels)}
            y_true = np.array(all_targets, dtype=int)

            # Build score matrix (n_samples, k)
            y_score = np.zeros((total, k), dtype=float)

            for i, curr in enumerate(all_curr):
                counts = transition_counts.get(int(curr), None)

                if counts is None or sum(counts.values()) == 0:
                    # unseen current activity -> uniform over labels present in test
                    y_score[i, :] = 1.0 / k
                else:
                    row_sum = 0.0
                    for next_act, c in counts.items():
                        if next_act in label_to_col:
                            y_score[i, label_to_col[next_act]] = float(c)
                            row_sum += float(c)

                    if row_sum == 0.0:
                        y_score[i, :] = 1.0 / k
                    else:
                        y_score[i, :] /= row_sum

            try:
                if k == 2:
                    # binary AUROC expects score for "positive" class
                    pos_label = labels[1]
                    auroc_macro = roc_auc_score((y_true == pos_label).astype(int), y_score[:, label_to_col[pos_label]])
                else:
                    auroc_macro = roc_auc_score(y_true, y_score, labels=labels, multi_class="ovr", average="macro")
            except ValueError:
                auroc_macro = float("nan")

    return accuracy, f1_macro, auroc_macro, correct, total



def main():
    args = parse_args()
    
    print("=" * 80)
    print("Baseline Model: Next Event Prediction via Transition Frequencies")
    print("=" * 80)
    
    config = {
        "log": args.dataset,
        "batch_size": 64,
        "categorical_features": ["activity"],
        "continuous_features": [],
        "categorical_targets": ["activity"],
        "continuous_targets": None,
        "val_size": 0.0,
        "val_split": "classic",
        "lifecycle": args.lifecycle,
        "num_workers": 0,
    }
    
    train_log, train_loader, test_loader, val_loader, dataset_info = charge_loaders(config)
    
    print("\nBuilding transition table from training data...")
    transition_counts = build_transition_counts(train_loader)
    predictions = build_predictions(transition_counts, args.seed)
    print(f"Transition table built with {len(predictions)} unique activities")
    
    print("\nEvaluating on test set...")
    accuracy, f1_macro, auroc_macro, correct, total = evaluate(test_loader, transition_counts, predictions)

    
    print(f"\nResults:")
    print(f"  Correct: {correct}")
    print(f"  Total: {total}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  AUROC Macro: {auroc_macro:.4f}" if auroc_macro == auroc_macro else "  AUROC Macro: nan")

    
    if args.wandb and WANDB_AVAILABLE:
        login_wandb_from_env(wandb)
        itos = train_log.itos["activity"]
        
        wandb.init(project=args.project_name, config={
            "backbone": "baseline_transition_frequency",
            "log": args.dataset,
            "lifecycle": args.lifecycle,
            "seed": args.seed,
        })
        wandb.log({f"dataset/{k}": v for k, v in dataset_info.items()}, step=0)
        wandb.log({
            "best_test_final_next_activity_acc": accuracy,
            "best_test_final_next_activity_f1_macro": f1_macro,
            "best_test_final_next_activity_auroc_macro": auroc_macro,
        })
        
        transition_data = [
            [itos.get(act, str(act)), itos.get(next_act, str(next_act)), count]
            for act, counts in transition_counts.items()
            for next_act, count in counts.items()
        ]
        wandb.log({"transition_table": wandb.Table(
            columns=["current_activity", "next_activity", "count"],
            data=transition_data
        )})
        
        prediction_data = [
            [itos.get(act, str(act)), itos.get(pred, str(pred))]
            for act, pred in predictions.items()
        ]
        wandb.log({"prediction_table": wandb.Table(
            columns=["activity", "predicted_next"],
            data=prediction_data
        )})
        wandb.finish()
    
    print("=" * 80)
    return accuracy


if __name__ == "__main__":
    main()
