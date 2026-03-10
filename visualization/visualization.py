from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.interpolate import griddata
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from config.paths import get_paths


ALIASES = {
    "best_test_final_next_activity_acc": "Best Test Activity Accuracy",
    "best_test_final_next_activity_loss": "Best Test Activity Loss",
    "best_test_final_next_activity_f1": "Best Test Activity F1",
    "best_test_final_next_activity_auroc": "Best Test Activity AUROC",
    "train_next_activity_loss": "Train Activity Loss",
    "train_next_activity_acc": "Train Activity Acc",
    "val_next_activity_loss": "Val Activity Loss",
    "val_next_activity_acc": "Val Activity Acc",
    "test_next_activity_loss": "Test Activity Loss",
    "test_next_activity_acc": "Test Activity Acc",
    "duration_sec": "Training Time (s)",
    "best_epoch": "Best Epoch",
    "BPI20PrepaidTravelCosts": "BPI20-Prepaid",
    "BPI20RequestForPayment": "BPI20-Request",
    "BPI20TravelPermitData": "BPI20-Permit",
    "BPI12": "BPI12",
    "BPI17": "BPI17",
    "qwen25-05b": "Qwen2.5-0.5B",
    "llama32-1b": "LLaMA3.2-1B",
    "gpt2": "GPT-2",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large": "GPT-2 Large",
    "gpt2-xl": "GPT-2 XL",
    "distilgpt2": "DistilGPT-2",
    "gpt2-tiny": "GPT-2 Tiny",
    "rnn": "LSTM",
    "student_model": "Student",
}


def alias(value) -> str:
    if isinstance(value, str):
        return ALIASES.get(value, value)
    return str(value)


def apply_aliases(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(alias)
    return df


def _apply_style():
    sns.set_theme(style="ticks", font="serif", rc={
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


@dataclass
class PlotConfig:
    figsize: tuple[float, float] = (6, 4)
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    palette: str = "colorblind"
    legend_title: str | None = None
    xtick_rotation: int = 0


def _setup_plot(config: PlotConfig | None, ax=None):
    _apply_style()
    config = config or PlotConfig()
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.get_figure()
    return fig, ax, config


def _apply_labels(fig, ax, config: PlotConfig, metric: str | None = None, x_col: str | None = None):
    if config.xlabel:
        ax.set_xlabel(config.xlabel)
    elif x_col:
        ax.set_xlabel(alias(x_col))

    if config.ylabel:
        ax.set_ylabel(config.ylabel)
    elif metric:
        ax.set_ylabel(alias(metric))

    if config.title:
        ax.set_title(config.title)

    if config.xtick_rotation:
        ax.tick_params(axis="x", rotation=config.xtick_rotation)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    if ax.get_legend():
        legend = ax.get_legend()
        if config.legend_title:
            legend.set_title(config.legend_title)
        legend.set_frame_on(False)

    fig.tight_layout()
    return fig


def plot_learning_curves(
    runs: pd.DataFrame,
    history: pd.DataFrame,
    y: str,
    group_by: str | None = None,
    config: PlotConfig | None = None,
    ax=None,
):
    fig, ax, config = _setup_plot(config, ax)

    # Filter history to only runs present in the runs DataFrame
    run_ids = runs["id"].dropna().astype(str).tolist()
    hist_filtered = history[history["run_id"].isin(run_ids)].copy()

    if group_by:
        merged = hist_filtered.merge(runs[["id", group_by]].rename(columns={"id": "run_id"}), on="run_id", how="inner")
        merged = apply_aliases(merged, [group_by])
    else:
        merged = hist_filtered

    plot_df = merged[["step", y] + ([group_by] if group_by else [])].dropna()

    lineplot_kwargs = {"data": plot_df, "x": "step", "y": y, "ax": ax, "errorbar": "sd"}
    if group_by:
        lineplot_kwargs["hue"] = group_by
        lineplot_kwargs["palette"] = config.palette
    sns.lineplot(**lineplot_kwargs)

    ax.set_xlabel(config.xlabel or "Epoch")
    return _apply_labels(fig, ax, config, metric=y)


def plot_split_curves(
    history: pd.DataFrame,
    metrics: dict[str, str],
    config: PlotConfig | None = None,
    ax=None,
    baseline: float | None = None,
):
    fig, ax, config = _setup_plot(config, ax)

    plot_data = []
    labels_order = []
    for col, label in metrics.items():
        if col in history.columns:
            temp = history[["step", col]].dropna().copy()
            temp["Split"] = label
            temp = temp.rename(columns={col: "value"})
            plot_data.append(temp)
            labels_order.append(label)

    if not plot_data:
        return fig

    plot_df = pd.concat(plot_data, ignore_index=True)

    split_palette = {"Train": "#2563eb", "Validation": "#f97316", "Test": "#16a34a", "Baseline": "#000000"}
    palette = {k: split_palette.get(k, "#6b7280") for k in labels_order}

    sns.lineplot(
        data=plot_df,
        x="step",
        y="value",
        hue="Split",
        hue_order=labels_order,
        ax=ax,
        errorbar=None,
        palette=palette,
    )

    if baseline is not None:
        ax.axhline(baseline, color="#000000", linestyle="--", linewidth=1.5, label="Baseline")
        ax.legend()

    ax.set_xlabel(config.xlabel or "Epoch")
    return _apply_labels(fig, ax, config)


def plot_dataset_curves(
    runs: pd.DataFrame,
    history: pd.DataFrame,
    dataset: str,
    metrics: dict[str, str],
    baseline: float | None = None,
    config: PlotConfig | None = None,
    ax=None,
):
    fig, ax, config = _setup_plot(config, ax)

    run_ids = runs[runs["log"] == dataset]["id"].tolist()
    if not run_ids:
        ax.text(0.5, 0.5, f"No runs found for {dataset}", ha="center", va="center", transform=ax.transAxes)
        return fig

    dataset_history = history[history["run_id"].isin(run_ids)]

    plot_data = []
    labels_order = []
    for col, label in metrics.items():
        if col in dataset_history.columns:
            temp = dataset_history[["step", col]].dropna().copy()
            temp["Split"] = label
            temp = temp.rename(columns={col: "value"})
            plot_data.append(temp)
            labels_order.append(label)

    if not plot_data:
        ax.text(0.5, 0.5, f"No metric data for {dataset}", ha="center", va="center", transform=ax.transAxes)
        return fig

    plot_df = pd.concat(plot_data, ignore_index=True)

    split_palette = {"Train": "#2563eb", "Validation": "#f97316", "Test": "#16a34a"}
    palette = {k: split_palette.get(k, "#6b7280") for k in labels_order}

    sns.lineplot(
        data=plot_df,
        x="step",
        y="value",
        hue="Split",
        hue_order=labels_order,
        ax=ax,
        errorbar="sd",
        palette=palette,
    )

    if baseline is not None:
        ax.axhline(baseline, color="#000000", linestyle="--", linewidth=1.5, label="Baseline")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    n_runs = len(run_ids)
    ax.annotate(f"n={n_runs} runs", xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom", fontsize=8)

    ax.set_xlabel(config.xlabel or "Epoch")
    return _apply_labels(fig, ax, config)


def plot_combined_curves(
    history: pd.DataFrame,
    acc_metrics: dict[str, str],
    loss_metrics: dict[str, str],
    config: PlotConfig | None = None,
):
    _apply_style()
    config = config or PlotConfig(figsize=(12, 5))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize)

    split_palette = {"Train": "#2563eb", "Validation": "#f97316", "Test": "#16a34a"}

    for ax, metrics, title in [
        (ax1, acc_metrics, "Accuracy"),
        (ax2, loss_metrics, "Loss"),
    ]:
        plot_data = []
        labels_order = []
        for col, label in metrics.items():
            if col in history.columns:
                temp = history[["step", col]].dropna().copy()
                temp["Split"] = label
                temp = temp.rename(columns={col: "value"})
                plot_data.append(temp)
                labels_order.append(label)

        if plot_data:
            plot_df = pd.concat(plot_data, ignore_index=True)
            palette = {k: split_palette.get(k, "#6b7280") for k in labels_order}
            sns.lineplot(
                data=plot_df,
                x="step",
                y="value",
                hue="Split",
                hue_order=labels_order,
                ax=ax,
                errorbar=None,
                palette=palette,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title)
            legend = ax.get_legend()
            if legend:
                legend.set_frame_on(False)

    if config.title:
        fig.suptitle(config.title, y=1.02)

    fig.tight_layout()
    return fig


def plot_bar(
    df: pd.DataFrame,
    y: str,
    x: str,
    hue: str | None = None,
    config: PlotConfig | None = None,
    ax=None,
):
    fig, ax, config = _setup_plot(config, ax)

    plot_df = df[[x, y] + ([hue] if hue else [])].dropna()
    plot_df = apply_aliases(plot_df, [x] + ([hue] if hue else []))

    barplot_kwargs = {"data": plot_df, "x": x, "y": y, "ax": ax, "errorbar": "sd", "capsize": 0.1}
    if hue:
        barplot_kwargs["hue"] = hue
        barplot_kwargs["palette"] = config.palette
    sns.barplot(**barplot_kwargs)

    return _apply_labels(fig, ax, config, metric=y, x_col=x)


def plot_box(
    df: pd.DataFrame,
    y: str,
    x: str,
    hue: str | None = None,
    *,
    config: PlotConfig | None = None,
    ax=None,
    show_points: bool = True,
    point_alpha: float = 0.35,
):
    """Thesis-friendly distribution plot (box + optional seed dots).

    Use this when you want to show variability across runs (seeds).
    """
    fig, ax, config = _setup_plot(config, ax)

    plot_df = df[[x, y] + ([hue] if hue else [])].dropna()
    plot_df = apply_aliases(plot_df, [x] + ([hue] if hue else []))

    boxplot_kwargs = {
        "data": plot_df,
        "x": x,
        "y": y,
        "ax": ax,
        "showfliers": False,
    }
    if hue:
        boxplot_kwargs["hue"] = hue
        boxplot_kwargs["palette"] = config.palette
    sns.boxplot(**boxplot_kwargs)

    if show_points:
        strip_kwargs = {
            "data": plot_df,
            "x": x,
            "y": y,
            "ax": ax,
            "alpha": point_alpha,
            "size": 3,
            "legend": False,
        }
        if hue:
            strip_kwargs["hue"] = hue
            strip_kwargs["dodge"] = True
            strip_kwargs["palette"] = ["#333333"] * len(plot_df[hue].unique())
        else:
            strip_kwargs["color"] = "#333333"
        sns.stripplot(**strip_kwargs)

        # de-duplicate legend (box + strip add two)
        if ax.get_legend() and hue:
            handles, labels = ax.get_legend_handles_labels()
            uniq = []
            uniq_labels = []
            for h, l in zip(handles, labels):
                if l not in uniq_labels:
                    uniq.append(h)
                    uniq_labels.append(l)
            ax.legend(uniq[: len(set(plot_df[hue]))], uniq_labels[: len(set(plot_df[hue]))], frameon=False, title=config.legend_title)

    return _apply_labels(fig, ax, config, metric=y, x_col=x)


def plot_violin(
    df: pd.DataFrame,
    y: str,
    x: str,
    hue: str | None = None,
    *,
    config: PlotConfig | None = None,
    ax=None,
    show_points: bool = True,
    point_alpha: float = 0.35,
    cut: float = 0,
):
    """Violin plot with optional seed dots overlaid.

    Use this when you want to show the distribution shape across runs.
    """
    fig, ax, config = _setup_plot(config, ax)

    plot_df = df[[x, y] + ([hue] if hue else [])].dropna()
    plot_df = apply_aliases(plot_df, [x] + ([hue] if hue else []))

    violin_kwargs = {
        "data": plot_df,
        "x": x,
        "y": y,
        "ax": ax,
        "cut": cut,
        "inner": "quartile",
        "linewidth": 1,
    }
    if hue:
        violin_kwargs["hue"] = hue
        violin_kwargs["palette"] = config.palette
    sns.violinplot(**violin_kwargs)

    if show_points:
        strip_kwargs = {
            "data": plot_df,
            "x": x,
            "y": y,
            "ax": ax,
            "alpha": point_alpha,
            "size": 3,
            "legend": False,
        }
        if hue:
            strip_kwargs["hue"] = hue
            strip_kwargs["dodge"] = True
            strip_kwargs["palette"] = ["#333333"] * len(plot_df[hue].unique())
        else:
            strip_kwargs["color"] = "#333333"
        sns.stripplot(**strip_kwargs)

        # de-duplicate legend
        if ax.get_legend() and hue:
            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            uniq_h, uniq_l = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    uniq_h.append(h)
                    uniq_l.append(l)
            n_hue = len(plot_df[hue].unique())
            ax.legend(uniq_h[:n_hue], uniq_l[:n_hue], frameon=False, title=config.legend_title)

    return _apply_labels(fig, ax, config, metric=y, x_col=x)


def save_figure(fig, name: str, formats: list[str] | None = None, close: bool = True):
    path = get_paths().figures / name
    formats = formats or ["pdf", "png"]
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), format=fmt, bbox_inches="tight")
    if close:
        plt.close(fig)


#def plot_grouped_bars(
#    df: pd.DataFrame,
#    y: str,
#    x: str,
#    hue: str,
#    config: PlotConfig | None = None,
#    ax=None,
#):
#    fig, ax, config = _setup_plot(config, ax)

#    plot_df = df[[x, y, hue]].dropna()
#    plot_df = apply_aliases(plot_df, [x, hue])

#    sns.barplot(data=plot_df, x=x, y=y, hue=hue, ax=ax, errorbar="sd", capsize=0.1, palette=config.palette)

#    return _apply_labels(fig, ax, config, metric=y, x_col=x)


def plot_grouped_bars(
    df: pd.DataFrame,
    y: str,
    x: str,
    hue: str,
    *,
    x_order: list | None = None,
    hue_order: list | None = None,
    config: PlotConfig | None = None,
    ax=None,
    baseline: dict[str, float] | None = None,
):
    fig, ax, config = _setup_plot(config, ax)

    plot_df = df[[x, y, hue]].dropna()
    plot_df = apply_aliases(plot_df, [x, hue])

    sns.barplot(data=plot_df, x=x, y=y, hue=hue, ax=ax, errorbar="sd", capsize=0.05, palette=config.palette)

    baseline_handle = None
    if baseline:
        x_values = plot_df[x].unique()
        aliased_baseline = {alias(k): v for k, v in baseline.items()}
        for i, label in enumerate(x_values):
            val = aliased_baseline.get(label)
            if val is not None:
                line = ax.hlines(val, i - 0.4, i + 0.4, colors="black", linestyles="dotted", linewidth=2)
                if baseline_handle is None:
                    baseline_handle = line

    # Rebuild legend to include baseline and place it outside plot area
    handles, labels = ax.get_legend_handles_labels()
    if baseline_handle is not None:
        handles.append(baseline_handle)
        labels.append("Baseline")
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, title=config.legend_title)

    return _apply_labels(fig, ax, config, metric=y, x_col=x)


def _pvalue_to_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_correlation_bars(
    df: pd.DataFrame,
    group_col: str,
    metric_pairs: list[tuple[str, str]],
    config: PlotConfig | None = None,
    method: str = "pearson",
):
    _apply_style()
    config = config or PlotConfig()

    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got '{method}'")

    groups = df[group_col].dropna().unique()

    records = []
    for group in groups:
        group_df = df[df[group_col] == group]
        for m1, m2 in metric_pairs:
            if m1 in group_df.columns and m2 in group_df.columns:
                valid = group_df[[m1, m2]].dropna()
                n = len(valid)
                if n > 2 and HAS_SCIPY:
                    x, y = valid[m1].values, valid[m2].values
                    if method == "spearman":
                        corr, pval = spearmanr(x, y)
                    else:
                        corr, pval = pearsonr(x, y)
                elif n > 1:
                    corr = valid[m1].corr(valid[m2])
                    pval = np.nan
                else:
                    corr, pval, n = np.nan, np.nan, 0
                split1 = alias(m1).split()[0]
                split2 = alias(m2).split()[0]
                pair_label = f"{split1} vs {split2}"
                records.append({
                    "Dataset": alias(group),
                    "Pair": pair_label,
                    "Correlation": corr,
                    "pval": pval,
                    "n": n,
                })

    corr_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=config.figsize)
    sns.barplot(data=corr_df, x="Dataset", y="Correlation", hue="Pair", ax=ax, palette=config.palette)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(-1.15, 1.15)

    datasets = corr_df["Dataset"].unique()
    pairs = corr_df["Pair"].unique()
    n_pairs = len(pairs)
    bar_width = 0.8 / n_pairs

    for i, dataset in enumerate(datasets):
        for j, pair in enumerate(pairs):
            row = corr_df[(corr_df["Dataset"] == dataset) & (corr_df["Pair"] == pair)]
            if row.empty:
                continue
            corr_val = row["Correlation"].values[0]
            pval = row["pval"].values[0]
            n = row["n"].values[0]
            stars = _pvalue_to_stars(pval)

            x_pos = i + (j - (n_pairs - 1) / 2) * bar_width
            y_pos = corr_val + (0.08 if corr_val >= 0 else -0.12)
            label = f"n={n}"
            if stars:
                label += f" {stars}"
            ax.annotate(label, xy=(x_pos, y_pos), ha="center", va="bottom" if corr_val >= 0 else "top", fontsize=7)

    method_label = "Spearman" if method == "spearman" else "Pearson"
    title = config.title or f"{method_label} Correlation"
    ax.set_title(title)
    if config.xlabel:
        ax.set_xlabel(config.xlabel)
    if config.ylabel:
        ax.set_ylabel(config.ylabel)

    if config.xtick_rotation:
        ax.tick_params(axis="x", rotation=config.xtick_rotation)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    legend = ax.get_legend()
    if legend:
        if config.legend_title:
            legend.set_title(config.legend_title)
        legend.set_frame_on(False)

    fig.text(0.99, 0.01, "* p<0.05  ** p<0.01  *** p<0.001", ha="right", va="bottom", fontsize=7, style="italic")

    fig.tight_layout()
    return fig


def plot_3d_manifold(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    color: str | None = None,
    config: PlotConfig | None = None,
    manifold: bool = True,
    scatter: bool = True,
):
    _apply_style()
    config = config or PlotConfig(figsize=(8, 6))

    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="3d")

    plot_df = df[[x, y, z] + ([color] if color else [])].dropna()

    x_vals = plot_df[x].values
    y_vals = plot_df[y].values
    z_vals = plot_df[z].values

    if manifold and len(plot_df) >= 4 and HAS_SCIPY:
        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()
        if x_range > 1e-10 and y_range > 1e-10:
            try:
                xi = np.linspace(x_vals.min(), x_vals.max(), 30)
                yi = np.linspace(y_vals.min(), y_vals.max(), 30)
                xi, yi = np.meshgrid(xi, yi)
                zi = griddata((x_vals, y_vals), z_vals, (xi, yi), method="cubic")
                ax.plot_surface(xi, yi, zi, alpha=0.5, cmap="viridis", edgecolor="none")
            except Exception:
                pass

    if scatter:
        if color and color in plot_df.columns:
            unique_colors = plot_df[color].unique()
            cmap = plt.cm.get_cmap(config.palette if config.palette != "colorblind" else "tab10")
            for i, c in enumerate(unique_colors):
                mask = plot_df[color] == c
                ax.scatter(
                    x_vals[mask], y_vals[mask], z_vals[mask],
                    label=alias(c), c=[cmap(i / len(unique_colors))], s=50, edgecolors="k", linewidth=0.5
                )
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        else:
            ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap="viridis", s=50, edgecolors="k", linewidth=0.5)

    ax.set_xlabel(config.xlabel or alias(x))
    ax.set_ylabel(config.ylabel or alias(y))
    ax.set_zlabel(alias(z))

    if config.title:
        ax.set_title(config.title)

    fig.tight_layout()
    return fig
