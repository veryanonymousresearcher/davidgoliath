import torch

from contextlib import nullcontext
from pathlib import Path
from typing import Union, Tuple

from config.paths import get_paths
import gc
import torch._dynamo



from contextlib import nullcontext

_AUTODTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

def autocast_ctx(device: str, precision: str):
    dtype = _AUTODTYPE.get(precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def maybe_make_grad_scaler(precision: str):
    if precision == "fp16" and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=False)


def make_attention_mask_and_sanitize_targets(
    x_cat: torch.Tensor,
    y_cat: torch.Tensor,
    has_cat_targets: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attention_mask = (x_cat[..., 0] != 0).long()
    if has_cat_targets:
        y_for_mask = y_cat[..., 0] if y_cat.ndim == 3 else y_cat
        attention_mask = attention_mask * (y_for_mask != -1).long()
        y_cat = y_cat.clone()
        y_cat[y_cat == -1] = 0
    return attention_mask, y_cat


def make_attention_and_loss_masks_and_sanitize_targets(
    x_cat: torch.Tensor,
    y_cat: torch.Tensor,
    has_cat_targets: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build separate masks for model attention and loss computation.
    - attention_mask: pads only
    - loss_mask: pads + invalid targets (e.g. prefix-split masked positions)
    """
    attention_mask = (x_cat[..., 0] != 0).long()
    loss_mask = attention_mask
    if has_cat_targets:
        y_for_mask = y_cat[..., 0] if y_cat.ndim == 3 else y_cat
        loss_mask = attention_mask * (y_for_mask != -1).long()
        y_cat = y_cat.clone()
        y_cat[y_cat == -1] = 0
    return attention_mask, loss_mask, y_cat








def save_checkpoint(checkpoint: dict, experiment_id: Union[str, int]) -> Path:
    paths = get_paths()
    save_path = paths.model_path("suffix") / f"{experiment_id}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(ckpt_path: str, map_location=None, trust_checkpoint: bool = True):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_path (str): Path to checkpoint file
        map_location: Can be used to directly load to specific device
        trust_checkpoint (bool): If True, load with weights_only=False (can execute pickled code).
            Use True only if you trust the source of the checkpoint. If False, attempts a
            safe load by allow-listing known types and using weights_only=True.
    """
    # If we don't trust, try to allow-list known safe globals and load with weights_only=True
    if not trust_checkpoint:
        try:
            from torch.serialization import add_safe_globals
            try:
                from ppm.models.config import FreezeConfig
                add_safe_globals([FreezeConfig])
            except Exception:
                pass
            return torch.load(ckpt_path, map_location=map_location, weights_only=True)
        except Exception:
            # As a last resort (still safe_globals in place), try without weights_only
            return torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Trusted path: explicitly disable weights_only to avoid PyTorch 2.6 safe-loading errors
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        # For older PyTorch versions that don't support weights_only kwarg
        ckpt = torch.load(ckpt_path, map_location=map_location)
    print(" [*] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


def save_confidence_level(
    model,
    test_loader,
    config,
):
    import torch.nn.functional as F

    device = config["device"]

    n_classes = len(test_loader.dataset.log.itos["activity"])
    confidence_sum = torch.zeros(n_classes)
    total_count = torch.zeros(n_classes)
    total_count_ground_truth = torch.zeros(n_classes)
    accuracy_sum = torch.zeros(n_classes)

    def to_device(*args):
        return (item.to(device) for item in args)

    model.eval()
    with torch.inference_mode():
        for items in test_loader:
            x_cat, x_num, y_cat, y_num = to_device(*items)

            attention_mask = x_cat[..., 0] != 0

            outputs, _ = model(
                x_cat=x_cat, x_num=x_num, attention_mask=attention_mask.long()
            )

            mask_flat = attention_mask.view(-1)
            y_cat_flat = y_cat.view(-1)[mask_flat]

            for target in test_loader.dataset.log.targets.categorical:
                probs = F.softmax(outputs[target], dim=-1)
                max_probs, preds = probs.max(dim=-1)

                preds_flat = preds.view(-1)[mask_flat].cpu()
                probs_flat = max_probs.view(-1)[mask_flat].cpu()

                total_count += torch.bincount(preds_flat, minlength=n_classes)
                confidence_sum += torch.bincount(
                    preds_flat, weights=probs_flat, minlength=n_classes
                )
                total_count_ground_truth += torch.bincount(
                    y_cat_flat.cpu(), minlength=n_classes
                )

                accuracy_sum += torch.bincount(
                    preds_flat,
                    weights=(preds_flat == y_cat_flat.cpu()).float(),
                    minlength=n_classes,
                )

    avg_confidence = torch.where(
        total_count > 0, confidence_sum / total_count, torch.zeros(n_classes)
    )
    avg_accuracy = torch.where(
        total_count > 0, accuracy_sum / total_count, torch.zeros(n_classes)
    )

    import pandas as pd

    paths = get_paths()
    csv_path = paths.logs / "confidence_eval.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    new_rows = []
    for cls in range(n_classes):
        new_rows.append(
            {
                "log": config["log"],
                "backbone": config["backbone"],
                "activity": test_loader.dataset.log.itos["activity"][cls],
                "avg_confidence": avg_confidence[cls].item(),
                "avg_accuracy": avg_accuracy[cls].item(),
                "predicted_count": total_count[cls].item(),
                "true_count": total_count_ground_truth[cls].item(),
            }
        )

    results = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)
    results.to_csv(csv_path, index=False)


def overwrite_with_best_checkpoint(model, ckpt_path, device="cuda"):
    # If compiled, update the underlying original module (the one with real params)
    base_model = (
        model._orig_mod
        if isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
        else model
    )

    # (Optional but recommended if you're tight on VRAM) evict current weights from GPU first
    base_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load checkpoint ON CPU so we don't spike GPU memory
    ckpt = load_checkpoint(str(ckpt_path), map_location="cpu")

    # Overwrite weights in-place
    base_model.load_state_dict(ckpt["net"], strict=False)

    # Free the big checkpoint dict immediately
    del ckpt
    gc.collect()

    # If you are DONE training, you can also reset compile caches
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    # Move back to GPU for eval
    base_model.to(device).eval()
    return base_model
