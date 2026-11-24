"""Konfigurasi global, path, random seed, dan konstanta domain untuk UTS ML 2026."""

from __future__ import annotations

import os
from pathlib import Path

# Root paths
SRC_DIR = Path(__file__).resolve().parent
KODE_DIR = SRC_DIR.parent
REPO_ROOT = KODE_DIR.parent

# Data directory (dapat di-override lewat env var UTS_ML_DATA_DIR)
DATA_DIR = Path(os.environ.get("UTS_ML_DATA_DIR", REPO_ROOT / "data")).resolve()

# Output directories
RESULTS_DIR = KODE_DIR / "results"
FIGURES_DIR = KODE_DIR / "figures"
CACHE_DIR = KODE_DIR / "cache"

# File dataset input
REGRESI_TRAIN_PATH = DATA_DIR / "regresi_train.csv.gz"
REGRESI_TEST_PATH = DATA_DIR / "regresi_test.csv.gz"
KLASIFIKASI_TRAIN_PATH = DATA_DIR / "klasifikasi_train.csv.gz"
KLASIFIKASI_TEST_PATH = DATA_DIR / "klasifikasi_test.csv.gz"
CLUSTERING_PATH = DATA_DIR / "clustering.csv.gz"

# File submission output (di root repo sesuai spesifikasi UTS)
PRED_REGRESI_PATH = REPO_ROOT / "prediksi_regresi.csv"
PRED_KLASIFIKASI_PATH = REPO_ROOT / "prediksi_klasifikasi.csv"
PRED_CLUSTER_PATH = REPO_ROOT / "cluster.csv"

# Reproducibility
SEED = 42
N_JOBS = int(os.environ.get("UTS_ML_N_JOBS", max(1, (os.cpu_count() or 4) - 2)))

# Domain constants (dari spesifikasi UTS & kamus_data.md)
SNAPSHOT_DATE = "2025-08-31"
TEST_START_DATE = "2025-09-01"
TEST_END_DATE = "2025-10-31"
FRAUD_REVIEW_COST_IDR = 25_000.0


def siapkan_direktori() -> None:
    """Pastikan direktori output (results, figures, cache) tersedia."""
    for d in (RESULTS_DIR, FIGURES_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
