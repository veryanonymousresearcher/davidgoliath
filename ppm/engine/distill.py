import time
from contextlib import nullcontext
from typing import Dict, Tuple, Optional, Any, List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from ppm.engine.timing import TimingMeter
from ppm.engine.utils import save_checkpoint, load_checkpoint, autocast_ctx, maybe_make_grad_scaler, make_attention_and_loss_masks_and_sanitize_targets
from ppm.engine.token_losses import flatten_logits_and_targets, masked_ce_mean, masked_kl_mean

from config.paths import get_paths as _get_paths

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# TorchMetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC
from torchmetrics.aggregation import MeanMetric

torch.set_float32_matmul_precision("high")


class DistillTorchMetricsManager:
    """
    Metrics per (phase, target):
      - total_loss (MeanMetric)
      - ce_loss    (MeanMetric)
      - kl_loss    (MeanMetric)
      - acc        (MulticlassAccuracy)
      - f1         (MulticlassF1Score)
    """

    def __init__(
        self,
        device: str,
        acc_average: str = "micro",
        f1_average: str = "macro",
    ):
        self.device = device
        self.acc_average = acc_average
        self.f1_average = f1_average
        self._metrics: Dict[str, Any] = {}

    def _ensure_cat_metrics(self, phase: str, target: str, num_classes: int):
        for name in ["total_loss", "CE_loss", "KL_loss"]:
            k = f"{phase}_{target}_{name}"
            if k not in self._metrics:
                self._metrics[k] = MeanMetric().to(self.device)

        k_acc = f"{phase}_{target}_acc"
        k_f1 = f"{phase}_{target}_f1"

        if k_acc not in self._metrics:
            self._metrics[k_acc] = MulticlassAccuracy(
                num_classes=num_classes, average=self.acc_average
            ).to(self.device)

        if k_f1 not in self._metrics:
            self._metrics[k_f1] = MulticlassF1Score(
                num_classes=num_classes, average=self.f1_average
            ).to(self.device)

    def reset_phase(self, phase: str):
        for k, m in self._metrics.items():
            if k.startswith(f"{phase}_"):
                m.reset()

    @torch.no_grad()
    def update_batch(
        self,
        phase: str,
        target: str,
        student_logits: torch.Tensor,      # (B,T,C)
        y_cat: torch.Tensor,               # (B,T)
        attention_mask: torch.Tensor,      # (B,T)
        total_loss: torch.Tensor,          # scalar mean
        ce_loss: torch.Tensor,             # scalar mean
        kl_loss: torch.Tensor,             # scalar mean
    ):
        C = int(student_logits.size(-1))
        self._ensure_cat_metrics(phase, target, C)

        self._metrics[f"{phase}_{target}_total_loss"].update(total_loss.detach())
        self._metrics[f"{phase}_{target}_CE_loss"].update(ce_loss.detach())
        self._metrics[f"{phase}_{target}_KL_loss"].update(kl_loss.detach())

        logits_flat, y_flat = flatten_logits_and_targets(student_logits, y_cat, attention_mask)
        if logits_flat.numel() != 0:
            preds_flat = torch.argmax(logits_flat, dim=-1)
            self._metrics[f"{phase}_{target}_acc"].update(preds_flat, y_flat)
            self._metrics[f"{phase}_{target}_f1"].update(preds_flat, y_flat)

    def compute_phase(self, phase: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, m in self._metrics.items():
            if k.startswith(f"{phase}_"):
                v = m.compute()
                out[k] = float(v.item()) if torch.is_tensor(v) and v.numel() == 1 else float(v)
        return out


def _student_block_size(student_model: Module) -> Optional[int]:
    """
    Returns block_size for the distilled student backbone when available.
    """
    backbone = getattr(student_model, "backbone", None)
    config = getattr(backbone, "config", None)
    block_size = getattr(config, "block_size", None)
    return int(block_size) if block_size is not None else None


def _truncate_batch_to_block_size(
    x_cat: torch.Tensor,
    x_num: torch.Tensor,
    y_cat: torch.Tensor,
    block_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keep the most recent `block_size` time steps so positional ids stay in-range.
    """
    if block_size is None or x_cat.ndim < 2:
        return x_cat, x_num, y_cat

    seq_len = int(x_cat.size(1))
    if seq_len <= block_size:
        return x_cat, x_num, y_cat

    x_cat = x_cat[:, -block_size:, ...]
    if x_num.ndim >= 2:
        x_num = x_num[:, -block_size:, ...]
    if y_cat.ndim >= 2:
        y_cat = y_cat[:, -block_size:, ...]
    return x_cat, x_num, y_cat


def _sanitize_x_cat_for_model(x_cat: torch.Tensor, model: Module) -> torch.Tensor:
    """
    Clamp out-of-range categorical ids to UNK (1) for the columns used by `model`.
    This prevents embedding gather OOB when ids slip through due vocab/config mismatch.
    """
    if not hasattr(model, "in_layer") or not hasattr(model, "categorical_cols"):
        return x_cat

    if x_cat.ndim != 3 or x_cat.size(-1) == 0:
        return x_cat

    x_cat = x_cat.clone()
    for ix, col in enumerate(model.categorical_cols):
        if ix >= x_cat.size(-1):
            break
        if col not in model.in_layer.embedding_layers:
            continue

        emb = model.in_layer.embedding_layers[col]
        num_embeddings = int(emb.num_embeddings)
        vals = x_cat[..., ix]
        bad = (vals < 0) | (vals >= num_embeddings)
        if bad.any():
            vals = vals.clone()
            vals[bad] = 1  # UNK token
            x_cat[..., ix] = vals

    return x_cat


# -------------------------
# steps
# -------------------------
def distillation_step(
    teacher_model: Module,
    student_model: Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    metrics: DistillTorchMetricsManager,
    device: str = "cuda",
    scaler=None,
    grad_clip=None,
    precision: str = "bf16",
    timer: Optional[TimingMeter] = None,
    oom_batches: int = 0,
    epoch: int = 1,
    nr_epochs: int = 10,
    temperature: float = 2.0,
    alpha_start: float = 0.9,
    alpha_end: float = 0.1,
):
    """One training epoch for distillation."""
    student_model.train()
    teacher_model.eval()

    phase = "train"
    metrics.reset_phase(phase)

    # Schedule alpha across epochs
    progress = float(epoch) / float(max(1, nr_epochs))
    alpha = alpha_start + (alpha_end - alpha_start) * progress

    ac = autocast_ctx(device, precision)
    scaler = scaler if scaler is not None else maybe_make_grad_scaler(precision)

    from ppm.data_preparation.datafetcher import DataPrefetcher
    prefetcher = DataPrefetcher(data_loader, device)

    target_name = "next_activity"
    block_size = _student_block_size(student_model)
    truncation_warned = False

    while True:
        batch_start = time.time()
        data = prefetcher.next()
        if data is None:
            break

        x_cat, x_num, y_cat, _ = data
        seq_len_before = int(x_cat.size(1)) if x_cat.ndim >= 2 else None
        x_cat, x_num, y_cat = _truncate_batch_to_block_size(x_cat, x_num, y_cat, block_size)
        if (
            not truncation_warned
            and block_size is not None
            and seq_len_before is not None
            and seq_len_before > block_size
        ):
            print(
                f"[distill] Truncating sequence length from {seq_len_before} to student context window "
                f"{block_size}. Increase --context_window to at least {seq_len_before} to avoid truncation."
            )
            truncation_warned = True
        # Defend against occasional OOB categorical ids before embedding lookups.
        x_cat = _sanitize_x_cat_for_model(x_cat, student_model)
        x_cat = _sanitize_x_cat_for_model(x_cat, teacher_model)

        attention_mask, loss_mask, y_cat = make_attention_and_loss_masks_and_sanitize_targets(
            x_cat=x_cat,
            y_cat=y_cat,
            has_cat_targets=True,
        )
        y_activity = y_cat[..., 0] if y_cat.ndim == 3 else y_cat

        optimizer.zero_grad(set_to_none=True)

        with ac:
            with torch.inference_mode():
                t_out, _ = teacher_model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
                t_logits = t_out[target_name]  # (B,T,C)

            s_out, _ = student_model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
            s_logits = s_out[target_name]  # (B,T,C)

            ce_loss = masked_ce_mean(s_logits, y_activity, loss_mask)
            kl_loss = masked_kl_mean(t_logits, s_logits, loss_mask, temperature=temperature)
            total_loss = alpha * kl_loss + (1.0 - alpha) * ce_loss

        try:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(student_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip:
                    clip_grad_norm_(student_model.parameters(), grad_clip)
                optimizer.step()

        except torch.cuda.OutOfMemoryError:
            oom_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        metrics.update_batch(
            phase=phase,
            target=target_name,
            student_logits=s_logits,
            y_cat=y_activity,
            attention_mask=loss_mask,
            total_loss=total_loss,
            ce_loss=ce_loss,
            kl_loss=kl_loss,
        )

        if timer and torch.cuda.is_available():
            torch.cuda.synchronize()
            batch_end = time.time()
            timer.record_batch(batch_end - batch_start, batch_end)

    return metrics.compute_phase(phase), oom_batches


def eval_step(
    student_model: Module,
    data_loader: DataLoader,
    metrics: DistillTorchMetricsManager,
    precision: str = "bf16",
    device: str = "cuda",
    data_subset: str = "val",
):
    """Eval: only CE loss is used; KL is logged as 0.0 for consistent keys."""
    student_model.eval()

    phase = data_subset
    metrics.reset_phase(phase)

    ac = autocast_ctx(device, precision)

    from ppm.data_preparation.datafetcher import DataPrefetcher
    prefetcher = DataPrefetcher(data_loader, device)

    target_name = "next_activity"
    block_size = _student_block_size(student_model)
    truncation_warned = False

    with torch.inference_mode():
        while True:
            data = prefetcher.next()
            if data is None:
                break

            x_cat, x_num, y_cat, _ = data
            seq_len_before = int(x_cat.size(1)) if x_cat.ndim >= 2 else None
            x_cat, x_num, y_cat = _truncate_batch_to_block_size(x_cat, x_num, y_cat, block_size)
            if (
                not truncation_warned
                and block_size is not None
                and seq_len_before is not None
                and seq_len_before > block_size
            ):
                print(
                    f"[distill] Truncating sequence length from {seq_len_before} to student context window "
                    f"{block_size}. Increase --context_window to at least {seq_len_before} to avoid truncation."
                )
                truncation_warned = True
            x_cat = _sanitize_x_cat_for_model(x_cat, student_model)

            attention_mask, loss_mask, y_cat = make_attention_and_loss_masks_and_sanitize_targets(
                x_cat=x_cat,
                y_cat=y_cat,
                has_cat_targets=True,
            )
            y_activity = y_cat[..., 0] if y_cat.ndim == 3 else y_cat

            with ac:
                s_out, _ = student_model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
                s_logits = s_out[target_name]

                ce_loss = masked_ce_mean(s_logits, y_activity, loss_mask)
                kl_loss = ce_loss.new_tensor(0.0)
                total_loss = ce_loss  # during eval, total_loss == CE

            metrics.update_batch(
                phase=phase,
                target=target_name,
                student_logits=s_logits,
                y_cat=y_activity,
                attention_mask=loss_mask,
                total_loss=total_loss,
                ce_loss=ce_loss,
                kl_loss=kl_loss,
            )

    return metrics.compute_phase(phase)


# -------------------------
# engine
# -------------------------
def distill_engine(
    teacher_model: Module,
    student_model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    config: dict,
    use_wandb: bool,
    append_run_info: bool,
    model_config: dict,
    dataset_info: dict = None,
):
    device = config["device"]
    precision = config.get("precision", "bf16")

    teacher_model.to(device)
    student_model.to(device)

    if torch.cuda.is_available() and config.get("compile", False):
        teacher_model = torch.compile(teacher_model.to(device))
        student_model = torch.compile(student_model.to(device))

    oom_batches = 0
    best_loss = float("inf")
    no_improvement = 0
    last_saved_experiment_id = None
    last_saved_ckpt_path = None
    best_test_metrics = None

    timer = TimingMeter()
    scaler = maybe_make_grad_scaler(precision)

    metrics = DistillTorchMetricsManager(
        device=device,
        acc_average=config.get("acc_average", "micro"),
        f1_average=config.get("f1_average", "macro"),
    )

    # Unique experiment ID per run — mirrors train_engine naming convention
    paths = _get_paths()
    base_experiment_id = "{}_{}".format(config["log"], config["backbone"])
    if WANDB_AVAILABLE and wandb.run is not None:
        base_experiment_id = f"{base_experiment_id}_run_{wandb.run.id}"

    for epoch in range(config["epochs"]):
        timer.reset_epoch()

        train_metrics, oom_batches = distillation_step(
            teacher_model=teacher_model,
            student_model=student_model,
            data_loader=train_loader,
            optimizer=optimizer,
            metrics=metrics,
            device=device,
            scaler=scaler,
            grad_clip=config.get("grad_clip", None),
            precision=precision,
            timer=timer,
            oom_batches=oom_batches,
            epoch=epoch,
            nr_epochs=config["epochs"],
            temperature=float(config.get("temperature", 2.0)),
            alpha_start=float(config.get("alpha_start", 0.9)),
            alpha_end=float(config.get("alpha_end", 0.1)),
        )

        val_metrics = eval_step(
            student_model=student_model,
            data_loader=val_loader,
            metrics=metrics,
            precision=precision,
            device=device,
            data_subset="val",
        )

        test_metrics = None
        try:
            test_metrics = eval_step(
                student_model=student_model,
                data_loader=test_loader,
                metrics=metrics,
                precision=precision,
                device=device,
                data_subset="test",
            )
        except Exception as e:
            print(f"Warning: Could not evaluate test set at epoch {epoch}: {e}")

        timer.print_epoch_summary(epoch)

        combined = {}
        combined.update(train_metrics)
        combined.update(val_metrics)
        if test_metrics:
            combined.update(test_metrics)

        print(
            f"Epoch {epoch}: "
            + " | ".join(f"{k}: {v:.4f}" for k, v in combined.items())
        )

        if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
            wandb.run.summary["oom_batch_count"] = oom_batches
            wandb.log(combined)

        # ===== EARLY STOPPING =====
        loss_key = "val_next_activity_total_loss"
        if loss_key not in combined:
            raise KeyError(
                f"Missing '{loss_key}' in metrics. "
                f"Available val keys: {[k for k in combined.keys() if k.startswith('val_')]}"
            )

        activity_loss = combined[loss_key]

        if activity_loss < best_loss - config["min_delta"]:
            no_improvement = 0
            best_loss = activity_loss

            # Extract state — handle torch.compile wrapper
            if isinstance(student_model, torch._dynamo.eval_frame.OptimizedModule):
                net_state = student_model._orig_mod.state_dict()
                model_class = student_model._orig_mod.__class__.__name__
                model_module = student_model._orig_mod.__class__.__module__
            else:
                net_state = student_model.state_dict()
                model_class = student_model.__class__.__name__
                model_module = student_model.__class__.__module__

            cpkt = {
                "epoch": epoch,
                "net": net_state,
                "optim": optimizer.state_dict(),
                "stoi": train_loader.dataset.log.stoi,
                "itos": train_loader.dataset.log.itos,
                "model_class": model_class,
                "model_module": model_module,
                "model_config": model_config,
            }

            last_saved_ckpt_path = paths.model_path("suffix") / f"{base_experiment_id}.pth"
            save_checkpoint(checkpoint=cpkt, experiment_id=base_experiment_id)
            last_saved_experiment_id = base_experiment_id

            if test_metrics:
                best_test_metrics = test_metrics.copy()
                print(f"✓ Saved checkpoint and recorded test metrics (val_loss={activity_loss:.4f})")

        else:
            no_improvement += 1
            if no_improvement >= config["patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    timer.print_total_summary()

    # ===== REPORT BEST TEST METRICS =====
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS (Best Checkpoint)")
    print("=" * 80 + "\n")

    if best_test_metrics is not None:
        print("Using test metrics from best checkpoint (stored in-memory):")
        for k, v in best_test_metrics.items():
            print(f"  test_final_{k.removeprefix('test_')}: {v:.4f}")
        print("=" * 80 + "\n")

        if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
            wandb.log({
                f"best_test_final_{k.removeprefix('test_')}": v
                for k, v in best_test_metrics.items()
            })
            wandb.run.summary.update({
                f"best_test_final_{k.removeprefix('test_')}": v
                for k, v in best_test_metrics.items()
            })

    else:
        print("⚠️  No in-memory test metrics. Attempting to load checkpoint and evaluate...\n")

        try:
            if last_saved_ckpt_path is None or not last_saved_ckpt_path.exists():
                print(f"ERROR: No checkpoint found at {last_saved_ckpt_path}")
            else:
                print(f"Loading checkpoint from: {last_saved_ckpt_path}")
                ckpt = load_checkpoint(str(last_saved_ckpt_path), map_location=device)

                target_model = (
                    student_model._orig_mod
                    if isinstance(student_model, torch._dynamo.eval_frame.OptimizedModule)
                    else student_model
                )

                # Validate shapes before loading
                mismatches = [
                    k for k, v in ckpt["net"].items()
                    if k in target_model.state_dict()
                    and v.shape != target_model.state_dict()[k].shape
                ]
                if mismatches:
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {len(mismatches)} params. "
                        f"First: {mismatches[0]}"
                    )

                target_model.load_state_dict(ckpt["net"], strict=True)
                target_model.to(device).eval()

                fallback_test_metrics = eval_step(
                    student_model=target_model,
                    data_loader=test_loader,
                    metrics=metrics,
                    precision=precision,
                    device=device,
                    data_subset="test_final",
                )

                print("\nTest metrics from loaded checkpoint:")
                for k, v in fallback_test_metrics.items():
                    print(f"  {k}: {v:.4f}")
                print("=" * 80 + "\n")

                if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
                    wandb.log({f"best_{k}": v for k, v in fallback_test_metrics.items()})
                    wandb.run.summary.update({f"best_{k}": v for k, v in fallback_test_metrics.items()})

        except torch.cuda.OutOfMemoryError:
            print("\n❌ OOM while loading checkpoint. Run evaluation in a separate process.")
        except Exception as e:
            print(f"\n❌ Could not evaluate checkpoint: {e}")
            import traceback
            traceback.print_exc()

    optimizer.zero_grad(set_to_none=True)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.run.summary["oom_batch_count"] = oom_batches

    print("Distillation complete.")
