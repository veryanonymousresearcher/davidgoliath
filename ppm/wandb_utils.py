from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from config.paths import get_paths

WANDB_ENTITY = "bpm_llm"
BASELINE_PROJECT = "baseline-nep"


def fetch_wandb_experiment(project: str, force_update: bool = False) -> Path:
    db_path = get_paths().metrics / f"{project}.db"

    if db_path.exists() and not force_update:
        print(f"Database already exists: {db_path}")
        print("Use force_update=True to re-fetch from wandb")
        return db_path

    import wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{project}", filters={"state": "finished"})

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS runs")
        conn.execute("DROP TABLE IF EXISTS history")
        conn.execute(
            """CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                config TEXT,
                summary TEXT,
                metadata TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE history (
                run_id TEXT,
                step INTEGER,
                metrics TEXT,
                PRIMARY KEY (run_id, step),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )"""
        )

        run_count = 0
        history_count = 0

        for run in runs:
            config = dict(run.config)
            summary = {}
            for k, v in run.summary.items():
                if k == "_runtime":
                    summary["duration_sec"] = v
                elif not k.startswith(("_", "gradients/", "parameters/")):
                    summary[k] = v

            metadata = {
                "created_at": run.created_at,
                "tags": run.tags,
                "group": run.group,
                "job_type": run.job_type,
            }

            conn.execute(
                "INSERT INTO runs (id, name, config, summary, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.name,
                    json.dumps(config, default=str),
                    json.dumps(summary, default=str),
                    json.dumps(metadata, default=str),
                ),
            )
            run_count += 1

            try:
                history = run.scan_history()
                for row in history:
                    step = row.get("_step", 0)
                    metrics = {
                        k: v
                        for k, v in row.items()
                        if not k.startswith("_")
                        and not k.startswith("gradients/")
                        and not k.startswith("parameters/")
                    }
                    if metrics:
                        conn.execute(
                            "INSERT OR REPLACE INTO history (run_id, step, metrics) VALUES (?, ?, ?)",
                            (run.id, step, json.dumps(metrics, default=str)),
                        )
                        history_count += 1
            except Exception as e:
                print(f"Warning: Could not fetch history for run {run.id}: {e}")

        conn.commit()

    print(f"Fetched {run_count} runs and {history_count} history entries to {db_path}")
    return db_path


def load_experiment_data(project: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    db_path = get_paths().metrics / f"{project}.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"No local data found for project '{project}'. "
            f"Run fetch_wandb_experiment('{project}') first."
        )

    with sqlite3.connect(db_path) as conn:
        runs_raw = pd.read_sql("SELECT * FROM runs", conn)
        history_raw = pd.read_sql("SELECT * FROM history", conn)

    runs_records = []
    for _, row in runs_raw.iterrows():
        config = json.loads(row["config"]) if row["config"] else {}
        summary = json.loads(row["summary"]) if row["summary"] else {}
        metadata = json.loads(row["metadata"]) if "metadata" in row and row["metadata"] else {}
        record = {"id": row["id"], "name": row["name"], **config, **summary, **metadata}
        runs_records.append(record)
    runs_df = pd.DataFrame(runs_records)
    
    if "created_at" in runs_df.columns:
        runs_df["created_at"] = pd.to_datetime(runs_df["created_at"])

    history_records = []
    for _, row in history_raw.iterrows():
        metrics = json.loads(row["metrics"]) if row["metrics"] else {}
        record = {"run_id": row["run_id"], "step": row["step"], **metrics}
        history_records.append(record)
    history_df = pd.DataFrame(history_records)

    return runs_df, history_df


def list_local_experiments() -> list[str]:
    metrics_dir = get_paths().metrics
    if not metrics_dir.exists():
        return []
    return [p.stem for p in metrics_dir.glob("*.db")]


def delete_experiment_data(project: str) -> bool:
    db_path = get_paths().metrics / f"{project}.db"
    if db_path.exists():
        db_path.unlink()
        print(f"Deleted: {db_path}")
        return True
    print(f"No data found for project: {project}")
    return False


def fetch_baseline_tables(
    dataset: str, project: str = BASELINE_PROJECT, lifecycle: bool = False, force_update: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Gets the transition tables from the latest baseline run for this dataset."""
    lc_suffix = "_lifecycle" if lifecycle else ""
    cache_path = get_paths().metrics / f"{project}_{dataset}{lc_suffix}_tables.db"
    
    if cache_path.exists() and not force_update:
        with sqlite3.connect(cache_path) as conn:
            transitions_df = pd.read_sql("SELECT * FROM transitions", conn)
            try:
                predictions_df = pd.read_sql("SELECT * FROM predictions", conn)
            except Exception:
                predictions_df = None
        return transitions_df, predictions_df
    
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{project}", filters={
        "config.log": dataset,
        "config.backbone": "baseline_transition_frequency",
        "config.lifecycle": lifecycle,
        "state": "finished",
    })
    
    runs_list = list(runs)
    if not runs_list:
        raise ValueError(f"No baseline run found for dataset '{dataset}' in project '{project}'")
    
    run = runs_list[0]
    
    transitions_df = None
    predictions_df = None
    
    for key in ["transition_table", "transitions"]:
        try:
            artifact = api.artifact(f"{WANDB_ENTITY}/{project}/run-{run.id}-{key}:latest")
            table = artifact.get(key)
            transitions_df = pd.DataFrame(table.data, columns=table.columns)
            break
        except Exception:
            continue
    
    for key in ["prediction_table", "predictions"]:
        try:
            artifact = api.artifact(f"{WANDB_ENTITY}/{project}/run-{run.id}-{key}:latest")
            table = artifact.get(key)
            predictions_df = pd.DataFrame(table.data, columns=table.columns)
            break
        except Exception:
            continue
    
    if transitions_df is None:
        raise ValueError(f"Could not fetch transition_table for dataset '{dataset}'")
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path) as conn:
        transitions_df.to_sql("transitions", conn, index=False, if_exists="replace")
        if predictions_df is not None:
            predictions_df.to_sql("predictions", conn, index=False, if_exists="replace")
    
    return transitions_df, predictions_df


def load_multiple_experiments(
    projects: list[str],
    force_update: bool = False,
    skip_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_runs = []
    all_history = []

    for proj in projects:
        try:
            fetch_wandb_experiment(proj, force_update=force_update)
            runs, history = load_experiment_data(proj)
            runs["project"] = proj
            history["project"] = proj
            all_runs.append(runs)
            all_history.append(history)
        except Exception as e:
            if skip_missing:
                print(f"Skipping {proj}: {e}")
            else:
                raise

    if not all_runs:
        raise ValueError("No experiments loaded")

    combined_runs = pd.concat(all_runs, ignore_index=True)
    combined_history = pd.concat(all_history, ignore_index=True)

    print(f"Loaded {len(combined_runs)} runs from {len(all_runs)} experiments")
    if "log" in combined_runs.columns:
        print(f"Datasets: {combined_runs['log'].unique().tolist()}")

    return combined_runs, combined_history
