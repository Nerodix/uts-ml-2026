"""Utilitas reproduksibilitas, metrik evaluasi UTS, pencatatan hasil, dan tema plot."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import SEED

TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "statsmodels",
    "imbalanced-learn",
    "shap",
    "optuna",
    "matplotlib",
    "seaborn",
    "joblib",
]


def set_global_seed(seed: int = SEED) -> None:
    """Kunci random seed Python dan NumPy untuk reproduksibilitas."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Hitung quantile (pinball) loss pada kuantil q in (0, 1)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    diff = y_t - y_p
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def p90_coverage(y_true: np.ndarray, y_pred_p90: np.ndarray) -> float:
    """Proporsi observasi aktual yang berada <= prediksi P90 (target 0.90)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred_p90, dtype=np.float64)
    return float(np.mean(y_t <= y_p))


def fraud_total_cost(
    y_true: np.ndarray,
    flagged: np.ndarray,
    nominal: np.ndarray,
    review_cost: float = 25_000.0,
) -> float:
    """Total biaya klasifikasi fraud sesuai rumus UTS.

    cost = sum(nominal fraud yang lolos) + review_cost * jumlah transaksi ditandai.
    """
    y_t = np.asarray(y_true, dtype=np.int8)
    f = np.asarray(flagged, dtype=np.int8)
    nom = np.asarray(nominal, dtype=np.float64)
    missed_fraud_loss = float(np.sum(nom[(y_t == 1) & (f == 0)]))
    review_total = float(review_cost * np.sum(f == 1))
    return missed_fraud_loss + review_total


def save_json(data: Any, path: Path) -> None:
    """Simpan dictionary/list ke JSON dengan format rapi dan deterministik."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        fh.write("\n")


def load_json(path: Path) -> Any:
    """Muat file JSON."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def collect_environment_info() -> dict[str, Any]:
    """Kumpulkan versi Python, seed, dan pustaka ML utama untuk reproduksibilitas."""
    versions: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return {
        "python": platform.python_version(),
        "seed": SEED,
        "packages": versions,
    }


def configure_plotting() -> None:
    """Set gaya grafik matplotlib/seaborn konsisten untuk laporan PDF."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
        }
    )


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Context manager untuk mencatat durasi eksekusi tiap tahap."""
    t0 = time.perf_counter()
    print(f"[START] {label}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[DONE]  {label} ({dt:.2f}s)")
