import time
import gc
from contextlib import nullcontext
from typing import Dict, Tuple, Optional, Any, List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from ppm.engine.utils import save_checkpoint, load_checkpoint, autocast_ctx, maybe_make_grad_scaler, make_attention_mask_and_sanitize_targets, make_attention_and_loss_masks_and_sanitize_targets, overwrite_with_best_checkpoint
from ppm.engine.timing import TimingMeter
from ppm.engine.memory import assess_training_footprint

from config.paths import get_paths as _get_paths

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# TorchMetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)
from torchmetrics.regression import MeanSquaredError
from torchmetrics.aggregation import MeanMetric

#fp8 training
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs


torch.set_float32_matmul_precision("high")


def _compute_batch_losses(
    out: Dict[str, torch.Tensor],
    y_cat: torch.Tensor,
    y_num: torch.Tensor,
    attention_mask: torch.Tensor,
    cat_targets: List[str],
    num_targets: List[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns:
      - total_loss: torch scalar (sum of per-target *mean* losses)
      - per_target_loss: dict {target_name: torch scalar mean loss}

    All losses are mean over valid tokens (no epoch-level division needed).
    """
    mask = attention_mask.bool().view(-1)

    some_key = next(iter(out.keys()))
    total_loss = out[some_key].new_tensor(0.0)
    per_target: Dict[str, torch.Tensor] = {}

    # Categorical: CE mean over valid positions
    for ix, target in enumerate(cat_targets):
        logits = out[target]  # (B,T,C)
        C = logits.size(-1)

        logits_flat = logits.view(-1, C)[mask]
        targets_flat = y_cat[..., ix].view(-1)[mask].long()

        if logits_flat.numel() == 0:
            per_target[target] = total_loss.new_tensor(0.0)
            continue

        loss_t = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
        per_target[target] = loss_t
        total_loss = total_loss + loss_t

    # Numerical: MSE mean over valid positions
    for ix, target in enumerate(num_targets):
        pred = out[target].view(-1)[mask]
        tgt = y_num[..., ix].view(-1)[mask]

        if pred.numel() == 0:
            per_target[target] = total_loss.new_tensor(0.0)
            continue

        loss_t = F.mse_loss(pred, tgt, reduction="mean")
        per_target[target] = loss_t
        total_loss = total_loss + loss_t

    return total_loss, per_target


class TorchMetricsManager:
    """
    TorchMetrics per (phase, target).
    Includes: acc/f1 (categorical), mse (numerical), and loss (both) + total loss per phase.
    """

    def __init__(
        self,
        device: str,
        f1_average: str = "macro",
        acc_average: str = "micro",
    ):
        self.device = device
        self.f1_average = f1_average
        self.acc_average = acc_average

        self._metrics: Dict[str, Any] = {}

    def _ensure_cat_metrics(self, phase: str, target: str, num_classes: int):
        acc_key = f"{phase}_{target}_acc"
        f1_key = f"{phase}_{target}_f1"
        loss_key = f"{phase}_{target}_loss"

        if acc_key not in self._metrics:
            self._metrics[acc_key] = MulticlassAccuracy(
                num_classes=num_classes, average=self.acc_average
            ).to(self.device)

        if f1_key not in self._metrics:
            self._metrics[f1_key] = MulticlassF1Score(
                num_classes=num_classes, average=self.f1_average
            ).to(self.device)

        if loss_key not in self._metrics:
            self._metrics[loss_key] = MeanMetric().to(self.device)

    def _ensure_num_metrics(self, phase: str, target: str):
        mse_key = f"{phase}_{target}_mse"
        loss_key = f"{phase}_{target}_loss"

        if mse_key not in self._metrics:
            self._metrics[mse_key] = MeanSquaredError().to(self.device)
        if loss_key not in self._metrics:
            self._metrics[loss_key] = MeanMetric().to(self.device)

    def _ensure_total_loss(self, phase: str):
        key = f"{phase}_loss"
        if key not in self._metrics:
            self._metrics[key] = MeanMetric().to(self.device)

    def reset_phase(self, phase: str):
        """Reset all metrics for a given phase."""
        for k, m in self._metrics.items():
            if k.startswith(f"{phase}_"):
                m.reset()

    @torch.no_grad()
    def update_batch(
        self,
        phase: str,
        out: Dict[str, torch.Tensor],
        y_cat: torch.Tensor,
        y_num: torch.Tensor,
        attention_mask: torch.Tensor,
        cat_targets: List[str],
        num_targets: List[str],
        total_loss: Optional[torch.Tensor] = None,
        per_target_loss: Optional[Dict[str, torch.Tensor]] = None,
    ):
        mask = attention_mask.bool().view(-1)

        # total loss (scalar mean already)
        if total_loss is not None:
            self._ensure_total_loss(phase)
            self._metrics[f"{phase}_loss"].update(total_loss.detach().float().cpu())

        # Categorical metrics
        for ix, target in enumerate(cat_targets):
            logits = out[target].detach()  # (B,T,C)
            C = int(logits.size(-1))
            self._ensure_cat_metrics(phase, target, C)

            logits_flat = logits.view(-1, C)[mask]
            targets_flat = y_cat[..., ix].view(-1)[mask].long()

            if logits_flat.numel() != 0:
                preds_flat = torch.argmax(logits_flat, dim=-1)
                preds_cpu = preds_flat.detach().cpu()
                targets_cpu = targets_flat.detach().cpu()

                self._metrics[f"{phase}_{target}_acc"].update(preds_cpu, targets_cpu)
                self._metrics[f"{phase}_{target}_f1"].update(preds_cpu, targets_cpu)

            # per-target loss logging
            if per_target_loss is not None and target in per_target_loss:
                self._metrics[f"{phase}_{target}_loss"].update(per_target_loss[target].detach().float().cpu())

        # Numerical metrics
        for ix, target in enumerate(num_targets):
            self._ensure_num_metrics(phase, target)

            pred_full = out[target].detach()  
            pred = pred_full.view(-1)[mask]
            tgt = y_num[..., ix].view(-1)[mask]

            if pred.numel() != 0:
                self._metrics[f"{phase}_{target}_mse"].update(
                    pred.detach().float().cpu(),
                    tgt.detach().float().cpu(),
                )
            if per_target_loss is not None and target in per_target_loss:
                self._metrics[f"{phase}_{target}_loss"].update(
                    per_target_loss[target].detach().float().cpu()
                )

    def compute_phase(self, phase: str) -> Dict[str, float]:
        """Compute all metrics for a given phase."""
        out: Dict[str, float] = {}
        for k, m in self._metrics.items():
            if k.startswith(f"{phase}_"):
                val = m.compute()
                out[k] = float(val.item()) if torch.is_tensor(val) and val.numel() == 1 else float(val)
        return out


def train_step(
    model: Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    metrics: TorchMetricsManager,
    device: str = "cuda",
    scaler=None,
    grad_clip=None,
    precision: str = "bf16",
    timer: Optional[TimingMeter] = None,
    oom_batches: int = 0,
    accelerator:Optional[Accelerator] = None,  #for fp8 training
    use_separate_attention_loss_masks: bool = False,
):
    """Execute one training epoch."""
    model.train()
    phase = "train"
    metrics.reset_phase(phase)

    ac = autocast_ctx(device, precision)
    scaler = scaler if scaler is not None else maybe_make_grad_scaler(precision)

    from ppm.data_preparation.datafetcher import DataPrefetcher

    prefetcher = DataPrefetcher(data_loader, device)

    while True:
        batch_start = time.time()
        data = prefetcher.next()
        if data is None:
            break

        x_cat, x_num, y_cat, y_num = data
        cat_targets = data_loader.dataset.log.targets.categorical
        num_targets = data_loader.dataset.log.targets.numerical

        if use_separate_attention_loss_masks:
            attention_mask, loss_mask, y_cat = make_attention_and_loss_masks_and_sanitize_targets(
                x_cat=x_cat, y_cat=y_cat, has_cat_targets=bool(cat_targets),
            )
        else:
            attention_mask, y_cat = make_attention_mask_and_sanitize_targets(
                x_cat=x_cat, y_cat=y_cat, has_cat_targets=bool(cat_targets),
            )
            loss_mask = attention_mask

        optimizer.zero_grad(set_to_none=True)

        if precision == "fp8":
            with accelerator.autocast():
                out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
                total_loss, per_target_loss = _compute_batch_losses(out=out,
                    y_cat=y_cat,
                    y_num=y_num,
                    attention_mask=loss_mask,
                    cat_targets=cat_targets,
                    num_targets=num_targets,
                )


            accelerator.backward(total_loss)

            if grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        else:
            with ac:
                out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
                total_loss, per_target_loss = _compute_batch_losses(out=out,
                        y_cat=y_cat,
                        y_num=y_num,
                        attention_mask=loss_mask,
                        cat_targets=cat_targets,
                        num_targets=num_targets,
                    )


            if scaler is not None and scaler.is_enabled():
                scaler.scale(total_loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        #except torch.cuda.OutOfMemoryError:
        #    oom_batches += 1
        #    optimizer.zero_grad(set_to_none=True)
        #    del out, total_loss, per_target_loss, x_cat, x_num, y_cat, y_num, attention_mask
        #    if torch.cuda.is_available():
        #        torch.cuda.empty_cache()
        #    continue

        metrics.update_batch(
            phase=phase,
            out=out,
            y_cat=y_cat,
            y_num=y_num,
            attention_mask=loss_mask,
            cat_targets=cat_targets,
            num_targets=num_targets,
            total_loss=total_loss,
            per_target_loss=per_target_loss,
        )
        
        del out, total_loss, per_target_loss, x_cat, x_num, y_cat, y_num, attention_mask, loss_mask

        if timer and torch.cuda.is_available():
            torch.cuda.synchronize()
            batch_end = time.time()
            timer.record_batch(batch_end - batch_start, batch_end)

    return metrics.compute_phase(phase), oom_batches


def eval_step(
    model: Module,
    data_loader: DataLoader,
    metrics: TorchMetricsManager,
    device: str = "cuda",
    data_subset: str = "val",
    precision: str = "bf16",
    use_separate_attention_loss_masks: bool = False,
):
    """Execute one evaluation epoch."""
    model.eval()
    phase = data_subset
    metrics.reset_phase(phase)

    ac = autocast_ctx(device, precision)
    
    with torch.inference_mode():
        for items in data_loader:
            x_cat, x_num, y_cat, y_num = items
            x_cat, x_num, y_cat, y_num = (
                x_cat.to(device, non_blocking=True),
                x_num.to(device, non_blocking=True),
                y_cat.to(device, non_blocking=True),
                y_num.to(device, non_blocking=True),
            )

            cat_targets = data_loader.dataset.log.targets.categorical
            num_targets = data_loader.dataset.log.targets.numerical

            if use_separate_attention_loss_masks:
                attention_mask, loss_mask, y_cat = make_attention_and_loss_masks_and_sanitize_targets(
                    x_cat=x_cat, y_cat=y_cat, has_cat_targets=bool(cat_targets),
                )
            else:
                attention_mask, y_cat = make_attention_mask_and_sanitize_targets(
                    x_cat=x_cat, y_cat=y_cat, has_cat_targets=bool(cat_targets),
                )
                loss_mask = attention_mask

            with ac:
                out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
                total_loss, per_target_loss = _compute_batch_losses(
                    out=out,
                    y_cat=y_cat,
                    y_num=y_num,
                    attention_mask=loss_mask,
                    cat_targets=cat_targets,
                    num_targets=num_targets,
                )

            metrics.update_batch(
                phase=phase,
                out=out,
                y_cat=y_cat,
                y_num=y_num,
                attention_mask=loss_mask,
                cat_targets=cat_targets,
                num_targets=num_targets,
                total_loss=total_loss,
                per_target_loss=per_target_loss,
            )
            
            del out, total_loss, per_target_loss, x_cat, x_num, y_cat, y_num, attention_mask, loss_mask

    return metrics.compute_phase(phase)

def train_engine(
    model: Module,
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
    
    import inspect, os
    print(">>> TRAIN_ENGINE ENTERED <<<")
    print("train_engine from:", inspect.getsourcefile(train_engine))
    print("cwd:", os.getcwd())
    print("first 200 chars of source:\n", inspect.getsource(train_engine)[:200])


    """Main training loop with test metrics saved in-memory alongside checkpoints."""
    real_model_class = model.__class__.__name__
    real_model_module = model.__class__.__module__

    device = config["device"]
    precision = config.get("precision", "bf16")

    accelerator = None
    
    if precision == "fp8":
        te_kwargs = TERecipeKwargs(
            fp8_format="HYBRID",
            amax_history_len=1024,
            amax_compute_algo="max",
        )
        accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[te_kwargs])
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )
    else:
        model.to(device)
        
    was_compiled = False
    if (
        torch.cuda.is_available()
        and config.get("compile", False)
        and precision != "fp8"
    ):
        model = torch.compile(model)
        was_compiled = True


    if torch.cuda.is_available() and config.get("memory_safety_margin", False):
        info = assess_training_footprint(
            model=model,
            train_loader=train_loader,
            device=device,
            precision=precision,
            optimizer=optimizer,
            safety_margin_frac=float(config.get("memory_safety_margin", 0.7)),
        )
        assert info["fits"], f"Batch too large. Suggest: {info['suggested_micro_batch_size']}"

    oom_batches = 0
    best_loss = float("inf")
    no_improvement = 0
    last_saved_experiment_id = None
    best_test_metrics = None  # In-memory storage for best test metrics

    timer = TimingMeter()
    scaler = maybe_make_grad_scaler(precision)

    metrics = TorchMetricsManager(
        device="cpu",
        f1_average=config.get("f1_average", "macro"),
        acc_average=config.get("acc_average", "micro"),
    )

    # Determine experiment ID and paths
    experiment_id = "{}_{}".format(config["log"], config["backbone"])
    paths = _get_paths()
    use_separate_attention_loss_masks = config.get("backbone") == "student_model"

    # ===== MAIN TRAINING LOOP =====
    for epoch in range(config["epochs"]):
        timer.reset_epoch()

        train_metrics, oom_batches = train_step(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            metrics=metrics,
            scaler=scaler,
            grad_clip=config.get("grad_clip", None),
            precision=precision,
            timer=timer,
            oom_batches=oom_batches,
            accelerator=accelerator,
            use_separate_attention_loss_masks=use_separate_attention_loss_masks,
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_metrics = eval_step(
            model=model,
            data_loader=val_loader,
            device=device,
            metrics=metrics,
            data_subset="val",
            precision=precision,
            use_separate_attention_loss_masks=use_separate_attention_loss_masks,
        )

        # Evaluate on test set (for logging/interest)
        test_metrics = None
        try:
            test_metrics = eval_step(
                model=model,
                data_loader=test_loader,
                device=device,
                metrics=metrics,
                data_subset="test",
                precision=precision,
                use_separate_attention_loss_masks=use_separate_attention_loss_masks,
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

        # ===== EARLY STOPPING LOGIC =====
        loss_key = "val_next_activity_loss"
        if loss_key not in combined:
            raise KeyError(
                f"Missing '{loss_key}' in metrics. "
                f"Available val keys: {[k for k in combined.keys() if k.startswith('val_')]}"
            )

        activity_loss = combined[loss_key]

        # Save checkpoint if improved
        print("Checking for improvement...")
        print(activity_loss, best_loss, config["min_delta"])
        if activity_loss < best_loss - config["min_delta"]:
            print("ok, here we go")
            if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
                net_state = model._orig_mod.state_dict()
            else:
                net_state = model.state_dict()

            cpkt = {
                "epoch": epoch,
                "net": net_state,
                "stoi": train_loader.dataset.log.stoi,
                "itos": train_loader.dataset.log.itos,
                "model_class": real_model_class,
                "model_module": real_model_module,
                "model_config": model_config,
            }

            final_experiment_id = experiment_id
            if append_run_info and WANDB_AVAILABLE and wandb.run is not None:
                val_loss_str = f"{activity_loss:.4f}".replace(".", "-")
                #final_experiment_id = f"{experiment_id}_run{wandb.run.id}_val-loss-{val_loss_str}"
                final_experiment_id = f"{experiment_id}_run_{wandb.run.id}"

            save_checkpoint(checkpoint=cpkt, experiment_id=final_experiment_id)
            last_saved_experiment_id = final_experiment_id

            # Overwrite best test metrics with current epoch's metrics
            if test_metrics:
                best_test_metrics = test_metrics.copy()
                print(f"✓ Saved checkpoint and recorded test metrics (val_loss={activity_loss:.4f})")
            print(f"Best test metrics: {best_test_metrics}")

        # Check for improvement
        if activity_loss < best_loss - config["min_delta"]:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= config["patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        best_loss = min(best_loss, activity_loss)

    timer.print_total_summary()

    # ===== REPORT BEST TEST METRICS =====
    print("\n" + "="*80)
    print("FINAL TEST RESULTS (Best Checkpoint)")
    print("="*80 + "\n")

    # Tier 1: Use in-memory metrics (preferred - no memory overhead)
    if best_test_metrics is not None:
        print("Using test metrics from best checkpoint (stored in-memory):")
        for k, v in best_test_metrics.items():
            print(f"  test_final_{k.removeprefix('test_')}: {v:.4f}")

        print("="*80 + "\n")

        if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
            wandb.log({
                f"best_test_final_{k.removeprefix('test_')}": v
                for k, v in best_test_metrics.items()
            })
            wandb.run.summary.update({
                f"best_test_final_{k.removeprefix('test_')}": v
                for k, v in best_test_metrics.items()
            })

    
    # Tier 2: Fallback - load checkpoint and evaluate (may OOM - that's acceptable)
    else:
        print("⚠️  No test metrics found in memory.")
        print("Attempting to load checkpoint and evaluate (may cause OOM)...\n")
        
        try:
            load_experiment_id = last_saved_experiment_id if last_saved_experiment_id else experiment_id
            ckpt_path = paths.model_path("suffix") / f"{load_experiment_id}.pth"
            
            if not ckpt_path.exists():
                print(f"ERROR: Checkpoint not found at {ckpt_path}")
                print("="*80 + "\n")
            else:
                print(f"Loading checkpoint from: {ckpt_path}")
                
                # Load checkpoint
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                
                # Load model state
                if was_compiled and isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
                    model._orig_mod.load_state_dict(checkpoint['net'])
                else:
                    model.load_state_dict(checkpoint['net'])
                
                del checkpoint
                gc.collect()
                
                # Evaluate on test set
                print("Running evaluation on test set...")
                fallback_test_metrics = eval_step(
                    model=model,
                    data_loader=test_loader,
                    device=device,
                    metrics=metrics,
                    data_subset="test",
                    precision=precision,
                    use_separate_attention_loss_masks=use_separate_attention_loss_masks,
                )
                
                print("\nTest metrics from loaded checkpoint:")
                for k, v in fallback_test_metrics.items():
                    print(f"  {k}: {v:.4f}")
                print("="*80 + "\n")
                
                if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
                    wandb.log({f"best_{k}": v for k, v in fallback_test_metrics.items()})
                    wandb.run.summary.update({f"best_{k}": v for k, v in fallback_test_metrics.items()})
                
        except torch.cuda.OutOfMemoryError:
            print("\n❌ OUT OF MEMORY while loading checkpoint.")
            print("The model is too large to reload after training in the same process.")
            print("Run evaluation in a separate script with a fresh Python process.")
            print("="*80 + "\n")
        except Exception as e:
            print(f"\n❌ ERROR: Could not evaluate checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.run.summary["oom_batch_count"] = oom_batches

    print("Training complete!")