# W&B Data Analysis

## Quick Start

```python
from ppm.wandb_utils import fetch_wandb_experiment, load_experiment_data

fetch_wandb_experiment("exp_001")
runs, history = load_experiment_data("exp_001")
```

### Multiple Experiments

```python
from ppm.wandb_utils import load_multiple_experiments

runs, history = load_multiple_experiments(
    ["exp_001", "exp_002", "exp_003"],
    force_update=False
)
```

## API Reference

### `fetch_wandb_experiment(project, force_update=False)`
Downloads runs from W&B and stores in `metrics/{project}.db`. Skips if already cached.

### `load_experiment_data(project)` → `(runs_df, history_df)`
- `runs`: One row per run with flattened config + summary
- `history`: One row per epoch per run

### `load_multiple_experiments(projects, force_update=False, skip_missing=True)` → `(runs_df, history_df)`
Loads and concatenates multiple experiments. Adds `project` column to track source.

### `list_local_experiments()` → `list[str]`
Returns cached project names.

### `delete_experiment_data(project)` → `bool`
Removes local cache.

## Plotting

```python
from visualization.visualization import (
    plot_bar, plot_learning_curves, plot_grouped_bars,
    plot_correlation_bars, plot_3d_manifold, save_figure, PlotConfig
)

plot_bar(runs, y="best_test_final_next_activity_acc", x="backbone")
plot_bar(runs, y="best_test_final_next_activity_acc", x="log", hue="backbone")
plot_grouped_bars(runs, y="best_test_final_next_activity_acc", x="lr_str", hue="log")
plot_correlation_bars(runs, group_col="log", metric_pairs=[("train_acc", "val_acc"), ...])
plot_3d_manifold(runs, x="lr", y="dataset_size", z="duration_sec", color="log")
plot_learning_curves(runs, history, y="val_next_activity_loss", group_by="backbone")
```

### PlotConfig

```python
config = PlotConfig(
    figsize=(8, 5),
    title="My Plot",
    xlabel="X Label",
    ylabel="Y Label",
    palette="Set2",
    legend_title="Legend",
    xtick_rotation=45
)
plot_bar(runs, y="acc", x="backbone", config=config)
```

### Save

```python
fig = plot_bar(runs, y="acc", x="backbone")
save_figure(fig, "my_plot")  # -> visualization/figures/my_plot.pdf, .png
```

## Available Functions

| Function | Description |
|----------|-------------|
| `plot_bar` | Bar chart (optional hue for grouping) |
| `plot_grouped_bars` | Bar chart with mandatory hue grouping |
| `plot_correlation_bars` | Pearson correlation between metric pairs per group |
| `plot_3d_manifold` | 3D scatter with optional interpolated surface |
| `plot_learning_curves` | Line plot of metrics over epochs |
| `save_figure` | Export to PDF/PNG |

## Available Columns

**Config:**
`log`, `backbone`, `lr`, `batch_size`, `epochs`, `fine_tuning`, `hidden_size`, `embedding_size`, `n_layers`, `rnn_type`, `strategy`, `lifecycle`, `lora_alpha`, `r`, `freeze_layers`, ...

**Summary:**
`best_test_final_next_activity_acc`, `best_test_final_next_activity_loss`, `train_next_activity_acc`, `val_next_activity_acc`, `test_next_activity_acc`, `duration_sec`, ...

**Metadata:**
`id`, `name`, `created_at`, `tags`, `group`, `project` (when using `load_multiple_experiments`)

## Database Structure

Each project is stored in `metrics/{project}.db` (SQLite):

```
├── runs (1 row per run)
│   ├── id          TEXT PRIMARY KEY - wandb run ID
│   ├── name        TEXT - run name ("ancient-wind-4")
│   ├── config      TEXT - JSON blob of training config
│   ├── summary     TEXT - JSON blob of final metrics
│   └── metadata    TEXT - JSON blob (created_at, tags, group)
│
└── history (1 row per epoch per run)
    ├── run_id      TEXT - foreign key to runs.id
    ├── step        INTEGER - epoch number
    └── metrics     TEXT - JSON blob of that epoch's metrics
```

When loaded via `load_experiment_data()`, JSON blobs are flattened into DataFrame columns.

## Disclaimer

This readme was generated with the help of AI.
