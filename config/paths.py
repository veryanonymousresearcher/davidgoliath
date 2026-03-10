import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Paths:
    root: Path = field(default_factory=lambda: Path.cwd())

    data: Path = field(default=None)
    models: Path = field(default=None)
    checkpoints: Path = field(default=None)
    cache: Path = field(default=None)
    logs: Path = field(default=None)
    metrics: Path = field(default=None)
    figures: Path = field(default=None)

    def __post_init__(self):
        if self.data is None:
            self.data = self.root / "data"
        if self.models is None:
            self.models = self.root / "persisted_models"
        if self.checkpoints is None:
            self.checkpoints = self.root / "distilled_checkpoints"
        if self.cache is None:
            self.cache = self.root / "cache"
        if self.logs is None:
            self.logs = self.root / "logs"
        if self.metrics is None:
            self.metrics = self.root / "metrics"
        if self.figures is None:
            self.figures = self.root / "visualization" / "figures"

    def dataset_path(self, name: str) -> Path:
        return self.data / name

    def cached_log_path(self, name: str) -> Path:
        return self.data / name / "cached_log.pkl"

    def cached_dataset_path(self, name: str, split: str) -> Path:
        return self.data / name / "cached_train_test" / f"{split}.pt"

    def model_path(self, subdir: str = "suffix") -> Path:
        return self.models / subdir

    def checkpoint_path(self, name: str) -> Path:
        return self.checkpoints / name

    def ensure_dirs(self):
        for path in [self.data, self.models, self.checkpoints, self.cache, self.logs, self.metrics, self.figures]:
            path.mkdir(parents=True, exist_ok=True)


_paths: Paths = None


def get_paths() -> Paths:
    global _paths
    if _paths is None:
        _paths = _init_paths()
    return _paths


def _init_paths() -> Paths:
    root = Path(os.getenv("PROJECT_ROOT", Path.cwd()))

    data = os.getenv("DATA_DIR")
    models = os.getenv("MODELS_DIR")
    checkpoints = os.getenv("CHECKPOINTS_DIR")
    cache = os.getenv("CACHE_DIR")
    logs = os.getenv("LOGS_DIR")
    metrics = os.getenv("METRICS_DIR")
    figures = os.getenv("FIGURES_DIR")

    return Paths(
        root=root,
        data=Path(data) if data else None,
        models=Path(models) if models else None,
        checkpoints=Path(checkpoints) if checkpoints else None,
        cache=Path(cache) if cache else None,
        logs=Path(logs) if logs else None,
        metrics=Path(metrics) if metrics else None,
        figures=Path(figures) if figures else None,
    )


def set_paths(paths: Paths):
    global _paths
    _paths = paths
