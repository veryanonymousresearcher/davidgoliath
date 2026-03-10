# ppm/engine/masked.py
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F


def flatten_logits_and_targets(
    logits: torch.Tensor,          # (B,T,C)
    targets: torch.Tensor,         # (B,T) or (B,T,1)
    attention_mask: torch.Tensor,  # (B,T)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens logits and targets over valid (masked) positions.
    """
    C = int(logits.size(-1))
    if targets.ndim == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    elif targets.ndim != 2:
        raise ValueError(
            f"Expected targets with shape (B,T) or (B,T,1), got {tuple(targets.shape)}."
        )

    if targets.shape[:2] != logits.shape[:2]:
        raise ValueError(
            f"Targets/logits shape mismatch: targets={tuple(targets.shape)}, logits={tuple(logits.shape)}."
        )
    if attention_mask.shape[:2] != logits.shape[:2]:
        raise ValueError(
            f"Mask/logits shape mismatch: mask={tuple(attention_mask.shape)}, logits={tuple(logits.shape)}."
        )

    mask = attention_mask.bool()
    valid_targets = (targets >= 0) & (targets < C)
    mask = (mask & valid_targets).reshape(-1)

    logits_all = logits.reshape(-1, C)
    targets_all = targets.reshape(-1).long()
    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    logits_flat = logits_all.index_select(0, valid_idx)
    targets_flat = targets_all.index_select(0, valid_idx)
    return logits_flat, targets_flat


def masked_ce_mean(
    logits: torch.Tensor,          # (B,T,C)
    targets: torch.Tensor,         # (B,T) or (B,T,1)
    attention_mask: torch.Tensor,  # (B,T)
) -> torch.Tensor:
    """
    Cross-entropy averaged over valid (masked) tokens.
    """
    logits_flat, targets_flat = flatten_logits_and_targets(logits, targets, attention_mask)
    if logits_flat.numel() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits_flat, targets_flat, reduction="mean")


def masked_kl_mean(
    teacher_logits: torch.Tensor,  # (B,T,C)
    student_logits: torch.Tensor,  # (B,T,C)
    attention_mask: torch.Tensor,  # (B,T)
    temperature: float,
) -> torch.Tensor:
    """
    KL( teacher || student ) with temperature, mean over valid tokens.
    Returns T^2 * mean_t KL, consistent with standard distillation scaling.
    """
    t = teacher_logits / temperature
    s = student_logits / temperature

    teacher_probs = torch.softmax(t, dim=-1)
    student_log_probs = torch.log_softmax(s, dim=-1)

    # (B,T,C) -> sum over classes -> (B,T)
    kl_per_class = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    kl_per_token = kl_per_class.sum(dim=-1)

    mask = attention_mask.bool()
    if mask.sum() == 0:
        return teacher_logits.new_tensor(0.0)

    return kl_per_token[mask].mean() * (temperature ** 2)
