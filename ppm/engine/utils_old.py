import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Union, Tuple

import torch

from config.paths import get_paths

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
    """
    fp16 -> GradScaler (CUDA only)
    bf16/fp8/cpu -> None
    Avoids torch.amp.GradScaler (missing in some torch builds).
    """
    if precision != "fp16":
        return None
    if not torch.cuda.is_available():
        return None
    from torch.cuda.amp import GradScaler
    return GradScaler(enabled=True)


def make_attention_mask_and_sanitize_targets(
    x_cat: torch.Tensor,
    y_cat: torch.Tensor,
    has_cat_targets: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attention_mask = (x_cat[..., 0] != 0).long()
    if has_cat_targets:
        attention_mask = attention_mask * (y_cat[..., 0] != -1).long()
        y_cat = y_cat.clone()
        y_cat[y_cat == -1] = 0
    return attention_mask, y_cat


def save_checkpoint(checkpoint: dict, experiment_id: Union[str, int]) -> Path:
    paths = get_paths()
    save_path = paths.model_path("suffix") / f"{experiment_id}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(ckpt_path: str, map_location=None, trust_checkpoint: bool = True):
    """Loads torch model from checkpoint file."""
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
            return torch.load(ckpt_path, map_location=map_location, weights_only=False)

    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=map_location)
    print(" [*] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


def _get_optimized_module_type():
    """
    Returns torch._dynamo.eval_frame.OptimizedModule type if available, else None.
    Import is lazy to avoid import-time issues across environments.
    """
    try:
        import torch._dynamo  # local import
        return torch._dynamo.eval_frame.OptimizedModule
    except Exception:
        return None


def overwrite_with_best_checkpoint(model, ckpt_path, device="cuda"):
    OptimizedModule = _get_optimized_module_type()

    base_model = (
        model._orig_mod
        if (OptimizedModule is not None and isinstance(model, OptimizedModule))
        else model
    )

    base_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ckpt = load_checkpoint(str(ckpt_path), map_location="cpu")
    base_model.load_state_dict(ckpt["net"], strict=False)

    del ckpt
    gc.collect()

    # Reset compile caches if available
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except Exception:
        pass

    base_model.to(device).eval()
    return base_model
