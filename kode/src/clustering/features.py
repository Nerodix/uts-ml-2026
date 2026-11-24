"""Pembentukan matriks fitur terpilih terbobot blok (26D) dan proyeksi PCA-12 Tugas 3b."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.clustering.preprocess import (  # noqa: E402
    INFORMATIVE_KATEGORI_COLUMNS,
    deteksi_anomali_bot_aktivitas,
    praproses_fitur_kanonik,
)
from src.config import SEED  # noqa: E402

# 13 fitur perilaku inti tertransformasi (bobot blok = 1,0)
SELECTED_CORE_BEHAVIOR_COLS: tuple[str, ...] = (
    "log_recency_hari",
    "log1p_frekuensi_90h",
    "log_rata2_keranjang",
    "log1p_item_per_pesanan",
    "log1p_retur_90h",
    "log1p_ulasan_90h",
    "log_sesi_per_minggu",
    "log_lama_member_bulan",
    "umur",
    "logit_persen_voucher",
    "logit_persen_flash_sale",
    "logit_persen_paylater",
    "logit_persen_cod",
)

# 3 fitur komposisional waktu belanja CLR independen (bobot blok = 1,0)
SELECTED_TIME_CLR_COLS: tuple[str, ...] = (
    "clr_jam_pagi",
    "clr_jam_siang",
    "clr_jam_malam",
)

# 1 fitur perangkat (bobot blok = 0,50 agar biner tidak memecah klaster buatan)
SELECTED_DEVICE_COLS: tuple[str, ...] = ("is_ios",)

# 9 fitur komposisional kategori informatif CLR (bobot blok = 0,45 agar tidak mendominasi perilaku inti)
SELECTED_CATEGORY_CLR_COLS: tuple[str, ...] = tuple(
    f"clr_{c}" for c in INFORMATIVE_KATEGORI_COLUMNS
)

SELECTED_26_COLS: tuple[str, ...] = (
    SELECTED_CORE_BEHAVIOR_COLS
    + SELECTED_TIME_CLR_COLS
    + SELECTED_DEVICE_COLS
    + SELECTED_CATEGORY_CLR_COLS
)

BLOCK_WEIGHTS: dict[str, float] = {
    "core_behavior": 1.00,
    "time_clr": 1.00,
    "device_ios": 0.50,
    "category_clr_top9": 0.45,
}

DEFAULT_N_PCA = 12


def build_clustering_features(
    df: pd.DataFrame,
    n_pca_components: int = DEFAULT_N_PCA,
    seed: int = SEED,
) -> dict[str, Any]:
    """Bangun matriks fitur terstandardisasi-terbobot (26D) dan proyeksi PCA-12 untuk pelanggan non-bot.

    Fungsi ini dipakai bersama oleh Tugas 3b–3g agar seluruh eksperimen model,
    pemilihan K, stabilitas, dan profil segmen bekerja pada ruang fitur yang persis sama.
    """
    is_bot, bot_summary = deteksi_anomali_bot_aktivitas(df, seed=seed)
    df_norm = df.loc[~is_bot].copy()
    df_canon, canon_meta = praproses_fitur_kanonik(df_norm)

    # Standardisasi tiap blok lalu terapkan bobot blok seimbang
    s_core = StandardScaler().fit_transform(
        df_canon[list(SELECTED_CORE_BEHAVIOR_COLS)].to_numpy(dtype=np.float64)
    ) * BLOCK_WEIGHTS["core_behavior"]
    s_time = StandardScaler().fit_transform(
        df_canon[list(SELECTED_TIME_CLR_COLS)].to_numpy(dtype=np.float64)
    ) * BLOCK_WEIGHTS["time_clr"]
    s_dev = StandardScaler().fit_transform(
        df_canon[list(SELECTED_DEVICE_COLS)].to_numpy(dtype=np.float64)
    ) * BLOCK_WEIGHTS["device_ios"]
    s_cat = StandardScaler().fit_transform(
        df_canon[list(SELECTED_CATEGORY_CLR_COLS)].to_numpy(dtype=np.float64)
    ) * BLOCK_WEIGHTS["category_clr_top9"]

    X_weighted_26d = np.hstack([s_core, s_time, s_dev, s_cat])
    pca = PCA(n_components=n_pca_components, random_state=seed)
    Z_pca = pca.fit_transform(X_weighted_26d)

    # Hitung pangsa varians tiap blok pada matriks 26D terbobot
    block_vars = {
        "core_behavior_13": float(np.sum(np.var(s_core, axis=0))),
        "time_clr_3": float(np.sum(np.var(s_time, axis=0))),
        "device_ios_1": float(np.sum(np.var(s_dev, axis=0))),
        "category_clr_9": float(np.sum(np.var(s_cat, axis=0))),
    }
    total_var = sum(block_vars.values())
    block_var_shares = {k: float(v / total_var * 100.0) for k, v in block_vars.items()}

    return {
        "is_bot": is_bot,
        "bot_summary": bot_summary,
        "df_normal": df_norm,
        "df_canon": df_canon,
        "canon_meta": canon_meta,
        "feature_names_26d": list(SELECTED_26_COLS),
        "X_weighted_26d": X_weighted_26d,
        "Z_pca": Z_pca,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pca_cumulative_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
        "block_weights": dict(BLOCK_WEIGHTS),
        "block_variance_share_pct": block_var_shares,
    }


