"""Seleksi fitur kuantitatif 59 kandidat & pembobotan blok Tugas 3b."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.clustering.preprocess import (  # noqa: E402
    ANONYMOUS_N_COLUMNS,
    ID_COLUMN,
    INFORMATIVE_KATEGORI_COLUMNS,
    LOW_SIGNAL_KATEGORI_COLUMNS,
    NOISE_NAMED_COLUMNS,
    PORSI_JAM_COLUMNS,
    PORSI_KATEGORI_COLUMNS,
    REDUNDANT_COLUMNS,
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


def _sarle_bimodality_coeff(x: np.ndarray) -> float:
    """Hitung koefisien bimodalitas Sarle; nilai > 0,555 mengindikasikan bimodalitas univariat."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 4:
        return 0.0
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr, fisher=True))  # excess kurtosis
    denom = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom <= 0.0:
        return 0.0
    return float((skew * skew + 1.0) / denom)


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


def _classify_raw_feature(col: str) -> str:
    """Kelompokkan status seleksi tiap kolom mentah untuk pelaporan transparan."""
    if col in ANONYMOUS_N_COLUMNS:
        return "dropped_anonymous_noise"
    if col in ("durasi_sesi_menit", "klik_iklan_30h", "wishlist"):
        return "dropped_bot_only_detector"
    if col in NOISE_NAMED_COLUMNS:
        return "dropped_named_noise"
    if col in REDUNDANT_COLUMNS:
        return "dropped_redundant"
    if col in LOW_SIGNAL_KATEGORI_COLUMNS:
        return "dropped_low_signal_category"
    if col in INFORMATIVE_KATEGORI_COLUMNS:
        return "selected_category_clr"
    if col in PORSI_JAM_COLUMNS:
        return "selected_time_clr"
    if col == "perangkat":
        return "selected_device_ios"
    return "selected_core_behavior"


def screen_all_features(df_normal: pd.DataFrame, seed: int = SEED) -> list[dict[str, Any]]:
    """Saring seluruh 59 fitur kandidat pada pelanggan non-bot dengan 3 metrik kuantitatif."""
    cand_cols = [c for c in df_normal.columns if c != ID_COLUMN]

    # Bangun tabel numerik peringkat untuk evaluasi koherensi & OOF Ridge R^2 pada subsampel 25k
    rng = np.random.default_rng(seed)
    idx_sub = rng.choice(len(df_normal), size=25_000, replace=False)
    sub = df_normal.iloc[idx_sub].copy()

    num_mat = pd.DataFrame(index=sub.index)
    for col in cand_cols:
        if col == "perangkat":
            num_mat[col] = (sub[col] == "ios").astype(np.float64)
        else:
            vals = sub[col].to_numpy(dtype=np.float64)
            if np.isnan(vals).any():
                vals = np.where(np.isnan(vals), float(np.nanmedian(vals)), vals)
            num_mat[col] = vals

    # 1. Matriks korelasi Spearman (pakai salinan writable untuk Pandas 3.x CoW)
    corr_df = num_mat.corr(method="spearman")
    corr_arr = corr_df.to_numpy(dtype=np.float64).copy()
    np.fill_diagonal(corr_arr, 0.0)

    # Basis pembanding koherensi: seluruh fitur non-anonim dan non-redundan
    ref_cols = [
        c
        for c in cand_cols
        if c not in ANONYMOUS_N_COLUMNS
        and c not in REDUNDANT_COLUMNS
        and c not in PORSI_KATEGORI_COLUMNS
        and c not in PORSI_JAM_COLUMNS
    ]
    ref_idx = [cand_cols.index(c) for c in ref_cols]

    # 2. Out-of-fold Ridge R^2 pada ruang rank-normalisasi (tahan skala & outlier Cauchy/Pareto!)
    rank_mat = np.column_stack(
        [stats.rankdata(num_mat[c].to_numpy(dtype=np.float64)) / (len(sub) + 1.0) for c in cand_cols]
    )
    rank_std = StandardScaler().fit_transform(rank_mat)

    # Gunakan fitur bernama bersih (non-anonim, non-redundan, bukan kolom target sendiri)
    clean_pred_base = [
        c
        for c in cand_cols
        if c not in ANONYMOUS_N_COLUMNS and c not in REDUNDANT_COLUMNS
    ]
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    rows: list[dict[str, Any]] = []
    for i, col in enumerate(cand_cols):
        # Spearman sum_r2 terhadap kolom perilaku acuan (kecuali dirinya sendiri)
        r_ref = np.array([corr_arr[i, j] for j in ref_idx if j != i], dtype=np.float64)
        sum_r2 = float(np.sum(r_ref * r_ref))
        # Korelasi terkuat terhadap semua kolom lain (termasuk redundansi)
        abs_all = np.abs(corr_arr[i])
        top_j = int(np.argmax(abs_all))
        max_abs_r = float(abs_all[top_j])
        top_partner = cand_cols[top_j]

        # OOF Ridge R^2 (kecualikan blok komposisional identik agar sum=1 tidak memicu R^2=1 trivial)
        exclude_set = {col}
        if col in PORSI_KATEGORI_COLUMNS:
            exclude_set.update(PORSI_KATEGORI_COLUMNS)
        elif col in PORSI_JAM_COLUMNS:
            exclude_set.update(PORSI_JAM_COLUMNS)
        pred_idx = [cand_cols.index(c) for c in clean_pred_base if c not in exclude_set]

        y = rank_std[:, i]
        Xp = rank_std[:, pred_idx]
        oof_pred = np.empty_like(y)
        for tr, va in kf.split(Xp):
            model = Ridge(alpha=50.0)
            model.fit(Xp[tr], y[tr])
            oof_pred[va] = model.predict(Xp[va])
        oof_r2 = float(max(0.0, r2_score(y, oof_pred)))

        # Sarle bimodality coefficient pada nilai mentah
        raw_vals = df_normal[col].to_numpy(dtype=np.float64) if col != "perangkat" else (
            df_normal[col] == "ios"
        ).to_numpy(dtype=np.float64)
        bc = _sarle_bimodality_coeff(raw_vals)

        rows.append(
            {
                "feature": col,
                "status": _classify_raw_feature(col),
                "spearman_sum_r2": round(sum_r2, 5),
                "max_abs_spearman": round(max_abs_r, 4),
                "top_partner": top_partner,
                "oof_ridge_r2": round(oof_r2, 5),
                "sarle_bimodality": round(bc, 4),
            }
        )

    rows.sort(key=lambda r: r["oof_ridge_r2"], reverse=True)
    return rows


