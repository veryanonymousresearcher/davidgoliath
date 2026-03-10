import torch
from contextlib import nullcontext


# ---------------------------------------------------------------------------
# Precision helper: selects autocast + scaler depending on precision setting
# ---------------------------------------------------------------------------
def _autocast_context(precision: str, device: str):
    """
    Returns:
        - autocast context (bf16/fp16/fp32)
        - GradScaler or None

    bf16 → BF16 autocast, no scaler
    fp16 → FP16 autocast + GradScaler
    fp32 → no autocast, no scaler
    """
    precision = (precision or "bf16").lower()
    if precision == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16), None
    if precision == "fp16":
        return torch.autocast(device_type=device, dtype=torch.float16), torch.cuda.amp.GradScaler()
    return nullcontext(), None



# ---------------------------------------------------------------------------
# Utility: sum real optimizer state memory (exact if state is populated)
# ---------------------------------------------------------------------------
def _optimizer_state_bytes(optimizer):
    """
    Computes the REAL memory footprint of optimizer.state tensors.
    Only works when optimizer.state is fully populated.
    """
    total = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state:
                continue
            for v in state.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
    return int(total)



# ---------------------------------------------------------------------------
# Fallback model for predicted optimizer state size (AdamW → 2× FP32, etc.)
# For sharded / offloaded / exotic optimizers where exact measurement fails.
# ---------------------------------------------------------------------------
def _predict_optimizer_state_bytes(model, optimizer):
    """
    Predict optimizer state memory from optimizer type when exact state
    allocation is not possible (e.g., ZeRO, FSDP-sharded, offload).

    This is the fallback for OPTION D.

    Default multipliers (very accurate for standard PyTorch):
      Adam/AdamW → 2× FP32 param memory
      SGD momentum → 1× FP32 param memory
      RMSprop → 1–2× FP32 param memory
      Adagrad → 1× FP32 param memory
      Unknown → 2× (safe default)
    """
    name = optimizer.__class__.__name__.lower()

    # Compute parameter memory
    param_bytes_fp32 = sum(p.numel() * 4 for p in model.parameters())  # FP32 size

    # Adam / AdamW
    if "adam" in name:
        return int(2 * param_bytes_fp32)

    # RMSProp
    if "rms" in name:
        has_momentum = any(g.get("momentum", 0) > 0 for g in optimizer.defaults.values())
        return int(param_bytes_fp32 * (2 if has_momentum else 1))

    # SGD
    if "sgd" in name:
        has_momentum = any(g.get("momentum", 0) > 0 for g in optimizer.defaults.values())
        return int(param_bytes_fp32 * (1 if has_momentum else 0))

    # Adagrad
    if "adagrad" in name:
        return int(param_bytes_fp32 * 1)

    # Unknown → assume AdamW-like (safe)
    return int(2 * param_bytes_fp32)



# ---------------------------------------------------------------------------
# FORCE optimizer state initialization (Option B)
# ---------------------------------------------------------------------------
def _ensure_optimizer_state_initialized(model, optimizer):
    """
    Try to force optimizer.state Materialization via a reversible dummy step.

    RETURNS:
        opt_bytes (int): real or predicted optimizer state memory
        used_fallback (bool): True if exact measurement failed
        reason (str): human-readable reason

    Logic (Option D):
        1) Try measuring real optimizer.state
        2) If empty → perform reversible dummy step
        3) If still empty or weird → fallback to predicted model
    """
    if optimizer is None:
        return 0, False, None

    # 1. If state is already populated, measure directly.
    any_state = any(optimizer.state.get(p) for g in optimizer.param_groups for p in g["params"])
    if any_state:
        return _optimizer_state_bytes(optimizer), False, None

    # 2. Try a reversible dummy step
    # Save parameters to restore later
    backups = [(p, p.data.clone()) for p in model.parameters() if p.requires_grad]

    try:
        # Assign fake grads so optimizer.step creates its state
        for g in optimizer.param_groups:
            for p in g["params"]:
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Measure exact state memory
        opt_bytes = _optimizer_state_bytes(optimizer)

        # If optimizer is sharded/offloaded → state may be zero or very small
        if opt_bytes == 0:
            pred = _predict_optimizer_state_bytes(model, optimizer)
            return pred, True, "optimizer state appears sharded/offloaded; using predicted size"

        return opt_bytes, False, None

    except Exception as e:
        # If dummy step failed (sharded/ZeRO/unsupported optimizer):
        pred = _predict_optimizer_state_bytes(model, optimizer)
        return pred, True, f"optimizer step failed ({e}); using predicted size"

    finally:
        # Restore parameters & cleanup grads
        for p, backup in backups:
            p.data.copy_(backup)
        optimizer.zero_grad(set_to_none=True)



# ---------------------------------------------------------------------------
# Dynamic activation / gradient memory probe
# ---------------------------------------------------------------------------
def _safe_forward_backward(model, x_cat, x_num, attention_mask, ac, scaler):
    """
    Measures dynamic peak memory used by forward+backward.
    Returns (delta_bytes, grad_params, oom_flag)
    """

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.memory_allocated()

    try:
        # Forward
        with ac:
            out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
            loss = (sum(v.float().sum() for v in out.values())
                    if isinstance(out, dict)
                    else out.float().sum())

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_params = sum(p.numel() for p in model.parameters() if p.grad is not None)

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        # Cleanup
        for p in model.parameters():
            p.grad = None

        return max(0, int(peak - start)), grad_params, False

    except RuntimeError as e:
        msg = str(e).lower()
        oom_like = (
            "out of memory" in msg
            or "cuda error 2" in msg
            or "cuda error: out of memory" in msg
            or "cuda out of memory" in msg
        )

        if oom_like:
            # Cleanup partially constructed graph
            for p in model.parameters():
                p.grad = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return None, None, True

        raise


# ---------------------------------------------------------------------------
# MAIN API: assess_training_footprint
# ---------------------------------------------------------------------------
def assess_training_footprint(
    model,
    train_loader,
    optimizer=None,
    device="cuda",
    precision: str = "bf16",
    safety_margin_frac: float = 0.9,
    print_report: bool = True,
):
    """
    Option D — best of both worlds:
      - dynamic activation+gradient probe
      - exact optimizer state memory if possible
      - fallback predicted optimizer state if sharded/offloaded
      - accounts for different precisions (bf16/fp16/fp32/fp8/tf32 kernels)
    """
    
# Count total + trainable params safely
    total_params = 0
    trainable_params = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n


    if not torch.cuda.is_available():
        if print_report:
            print("[MemoryProbe] CUDA unavailable.")
        return None

    # Take one batch
    try:
        sample = next(iter(train_loader))
    except StopIteration:
        if print_report:
            print("[MemoryProbe] train_loader empty.")
        return None

    # Inputs → device
    x_cat, x_num, *_ = sample
    x_cat, x_num = x_cat.to(device), x_num.to(device)
    attention_mask = (x_cat[..., 0] != 0).long()
    micro_bs = x_cat.shape[0]

    # Autocast / scaler
    ac, scaler = _autocast_context(precision, device)

    # Read free memory before probe
    free_before, device_total_bytes = torch.cuda.mem_get_info()
    safety_free = int(free_before * safety_margin_frac)

    # 1. Optimizer state memory (exact or fallback)
    opt_bytes, used_fallback, fallback_reason = _ensure_optimizer_state_initialized(model, optimizer)

    # 2. Dynamic activation/grad memory
    # 2. Dynamic activation/grad memory
    delta_bytes, grad_params, oom_flag = _safe_forward_backward(
        model, x_cat, x_num, attention_mask, ac, scaler
    )

    if oom_flag:
        if print_report:
            print("[MemoryProbe] Forward+backward OOMed on the sample batch. "
                "Try reducing micro-batch size, sequence length, or model size.")
        return None


    #print("Total parameters:", total)
    #print("Trainable parameters:", trainable)
    #print("Parameters that received gradients:", grad_params)
    #print(f"Fraction receiving grads: {grad_params / trainable:.2f}")


    # 3. Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # 4. Total estimated training-time peak memory
    total_training_bytes = delta_bytes + param_bytes + opt_bytes

    # 5. Activation per-sample
    per_sample_act = max(1, delta_bytes // max(1, micro_bs))

    # 6. Suggested micro-batch size
    suggested = max(1, safety_free // per_sample_act)

    info = dict(
        total_params=int(total_params),
        trainable_params=int(trainable_params),
        grad_params=int(grad_params),
        micro_batch_size=micro_bs,
        activation_peak_bytes=delta_bytes,
        activation_per_sample_bytes=per_sample_act,
        param_bytes=param_bytes,
        optimizer_state_bytes=opt_bytes,
        optimizer_state_fallback_used=used_fallback,
        optimizer_state_fallback_reason=fallback_reason,
        total_training_estimate_bytes=total_training_bytes,
        device_total_bytes=int(device_total_bytes),
        device_free_bytes=int(free_before),
        safety_free_bytes=int(safety_free),
        fits=bool(total_training_bytes <= safety_free),
        suggested_micro_batch_size=int(suggested),
    )

    if print_report:
        print(_pretty_memory_report(info, safety_margin_frac))

    return info



# ---------------------------------------------------------------------------
# Pretty report formatting
# ---------------------------------------------------------------------------
def _fmt(n):
    """Human-friendly bytes."""
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def _pretty_memory_report(info, safety):
    lines = []
    lines.append(f"Total parameters:                   {info['total_params']:,}")
    lines.append(f"Trainable parameters:               {info['trainable_params']:,}")
    lines.append(f"Parameters receiving gradients:     {info['grad_params']:,}")
    lines.append(
        f"Fraction receiving grads:           "
        f"{info['grad_params'] / max(1, info['trainable_params']):.2f}"
    )
    lines.append("")
    lines.append(f"Micro-batch size:                   {info['micro_batch_size']}")
    lines.append(f"Activation peak (measured):         {_fmt(info['activation_peak_bytes'])}")
    lines.append(f"Activation per-sample:              {_fmt(info['activation_per_sample_bytes'])}")
    lines.append("")
    lines.append(f"Parameter memory:                   {_fmt(info['param_bytes'])}")

    if info["optimizer_state_fallback_used"]:
        lines.append(f"Optimizer state memory (predicted): {_fmt(info['optimizer_state_bytes'])}")
        lines.append(f"  [!] Reason: {info['optimizer_state_fallback_reason']}")
    else:
        lines.append(f"Optimizer state memory (exact):     {_fmt(info['optimizer_state_bytes'])}")

    lines.append(f"Total training estimate:            {_fmt(info['total_training_estimate_bytes'])}")
    lines.append("")
    lines.append(f"GPU total:                          {_fmt(info['device_total_bytes'])}")
    lines.append(f"GPU free (before):                  {_fmt(info['device_free_bytes'])}")
    lines.append(f"Safety-free ({int(safety*100)}%):                  {_fmt(info['safety_free_bytes'])}")
    lines.append("")
    lines.append(f"Fits under safety-free memory:      {info['fits']}")
    lines.append(f"Suggested micro-batch size:         {info['suggested_micro_batch_size']}")
    return "\n".join(lines)

