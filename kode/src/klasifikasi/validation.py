"""Desain Validasi Berbasis Waktu Anti-Censoring untuk Tugas 2b (Klasifikasi Fraud).

Modul ini menyediakan skema validasi resmi Tugas 2 yang meniru pemakaian nyata
(memprediksi masa depan secara kronologis) sekaligus menangani masalah
*right-censoring* label akibat jeda konfirmasi fraud menjelang tanggal penarikan
data (31 Agustus 2025):

1. **Expanding-Window Time CV Matang (3 Fold: Mei, Jun, Jul 2025):**
   Seluruh fold validasi berjarak >= 31 hari sebelum snapshot 31 Agt 2025
   sehingga label 100% matang, memiliki riwayat latih minimal 4 bulan agar fitur
   perilaku pengguna/merchant (Tugas 2c) sudah hangat, dan berdurasi seragam
   1 bulan sehingga total biaya langsung sebanding antar fold.
2. **Perbandingan vs Stratified Random 5-Fold CV (Jan–Jul):**
   Membuktikan optimisme semu akibat mencampur transaksi dari waktu, pengguna,
   dan merchant yang sama antara latih dan validasi.
3. **Perbandingan vs Validasi Naif pada Agustus Tersensor (Agt 1–31 & Agt 15–31):**
   Membuktikan bahwa evaluasi pada jendela dengan label belum lengkap menurunkan
   PR-AUC secara artifisial (fraud yang belum terkonfirmasi dihukum sebagai
   *false positive*) dan mengacaukan rasio kalibrasi probabilitas.
4. **Split Evaluasi Pergeseran QRIS Agustus (``buat_split_shift_agustus``):**
   Menyediakan batas kronologis bersih (``Agt 1–10`` adaptasi vs ``Agt 11–21``
   evaluasi, membuang ekor tersensor berat ``Agt 22–31``) untuk Tugas 2e.

Jalankan langsung:
    python kode/src/klasifikasi/validation.py                  # Semua perbandingan + plot + JSON
    python kode/src/klasifikasi/validation.py --mode time      # Hanya Expanding-Window Time CV matang
    python kode/src/klasifikasi/validation.py --mode random    # Hanya Stratified Random 5-Fold CV
    python kode/src/klasifikasi/validation.py --mode censored  # Hanya evaluasi naif Agustus tersensor
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    FRAUD_REVIEW_COST_IDR,
    KLASIFIKASI_TRAIN_PATH,
    N_JOBS,
    RESULTS_DIR,
    SEED,
    siapkan_direktori,
)
from src.klasifikasi.preprocess import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    CLEAN_WITH_FLAGS_FEATURES,
    MATURE_TRAIN_END_DATE,
    SNAPSHOT_TIMESTAMP,
    TARGET_COLUMN,
    TIME_COLUMN,
    samakan_tipe_kategori,
    hitung_hari_sebelum_snapshot,
    tambah_flag_kualitas_data,
)
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    atur_gaya_grafik,
    hitung_biaya_total_fraud,
    label_batang,
    miring,
    tulis_json,
    atur_seed_global,
    sumbu_koma,
    catat_waktu,
)

# 3 Fold Expanding-Window Matang (100% di luar jendela right-censoring Agustus)
MATURE_TIME_FOLD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("Fold 1 (Mei)", "2025-04-30", "2025-05-01", "2025-05-31"),
    ("Fold 2 (Jun)", "2025-05-31", "2025-06-01", "2025-06-30"),
    ("Fold 3 (Jul)", "2025-06-30", "2025-07-01", "2025-07-31"),
)

# Jeda pengaman minimum (hari sebelum snapshot) agar label dianggap matang untuk validasi
MIN_MATURITY_LAG_DAYS = 25.0


@dataclass(frozen=True)
class FraudTimeFold:
    """Satu fold validasi berbasis waktu dengan batas tanggal eksplisit bebas sensor.

    Attributes
    ----------
    name : str
        Label ringkas fold (mis. ``'Fold 3 (Jul)'``).
    train_end : str
        Tanggal maksimum inklusif data latih (``YYYY-MM-DD``).
    val_start : str
        Tanggal minimum inklusif data validasi (``YYYY-MM-DD``).
    val_end : str
        Tanggal maksimum inklusif data validasi (``YYYY-MM-DD``).
    train_idx : np.ndarray
        Indeks posisi baris latih pada DataFrame sumber.
    val_idx : np.ndarray
        Indeks posisi baris validasi pada DataFrame sumber.
    """

    name: str
    train_end: str
    val_start: str
    val_end: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def filter_label_matang(
    df: pd.DataFrame,
    min_lag_hari: float = MIN_MATURITY_LAG_DAYS,
    snapshot_ts: pd.Timestamp = SNAPSHOT_TIMESTAMP,
) -> pd.Series:
    """Bangun masker boolean untuk baris yang labelnya telah melewati masa jeda konfirmasi.

    Parameters
    ----------
    df : pd.DataFrame
        Data transaksi yang memuat ``waktu`` atau ``waktu_dt``.
    min_lag_hari : float, default=25.0
        Jarak minimum (hari) dari waktu transaksi ke tanggal snapshot.
    snapshot_ts : pd.Timestamp, default=SNAPSHOT_TIMESTAMP
        Tanggal penarikan data latih (``2025-08-31 23:59:59``).

    Returns
    -------
    pd.Series
        Masker boolean ``True`` untuk transaksi dengan label matang.
    """
    waktu_dt = pd.to_datetime(df["waktu_dt"] if "waktu_dt" in df.columns else df[TIME_COLUMN])
    return hitung_hari_sebelum_snapshot(waktu_dt, snapshot_ts) >= float(min_lag_hari)


def buat_fold_waktu_matang(
    df: pd.DataFrame,
    specs: tuple[tuple[str, str, str, str], ...] = MATURE_TIME_FOLD_SPECS,
) -> list[FraudTimeFold]:
    """Bangun daftar fold Expanding-Window Time CV yang 100% bebas *right-censoring*.

    Menjamin tiga invarian anti-leakage pada setiap fold:
    1. Seluruh transaksi latih terjadi ketat sebelum awal jendela validasi.
    2. Tidak ada irisan baris antara latih dan validasi, maupun antar jendela validasi.
    3. Seluruh transaksi validasi berada pada periode label matang (<= 31 Jul 2025,
       jarak >= 31 hari sebelum snapshot 31 Agt 2025).

    Parameters
    ----------
    df : pd.DataFrame
        Data latih klasifikasi (urutan baris bebas; posisi diindeks lewat ``np.flatnonzero``).
    specs : tuple[tuple[str, str, str, str], ...]
        Daftar spesifikasi ``(nama, train_end, val_start, val_end)``.

    Returns
    -------
    list[FraudTimeFold]
        Daftar 3 fold validasi berbasis waktu matang.
    """
    waktu_dt = pd.to_datetime(df["waktu_dt"] if "waktu_dt" in df.columns else df[TIME_COLUMN])
    folds: list[FraudTimeFold] = []
    for name, tr_end, va_start, va_end in specs:
        t_tr_end = pd.Timestamp(f"{tr_end} 23:59:59")
        t_va_start = pd.Timestamp(f"{va_start} 00:00:00")
        t_va_end = pd.Timestamp(f"{va_end} 23:59:59")
        if t_tr_end >= t_va_start:
            raise ValueError(f"Fold {name}: batas latih ({tr_end}) harus sebelum validasi ({va_start}).")

        tr_mask = (waktu_dt <= t_tr_end).to_numpy()
        va_mask = ((waktu_dt >= t_va_start) & (waktu_dt <= t_va_end)).to_numpy()
        tr_idx = np.flatnonzero(tr_mask)
        va_idx = np.flatnonzero(va_mask)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise ValueError(f"Fold {name} kosong (train={len(tr_idx)}, val={len(va_idx)}).")
        folds.append(
            FraudTimeFold(
                name=name,
                train_end=tr_end,
                val_start=va_start,
                val_end=va_end,
                train_idx=tr_idx,
                val_idx=va_idx,
            )
        )
    return folds


def buat_split_shift_agustus(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Bangun indeks kronologis untuk mengevaluasi adaptasi gelombang QRIS Agustus (Tugas 2e).

    Memisahkan data latih menjadi:
    - ``jan_jul_idx``  : Jan 1 s.d. Jul 31 (matang 100%, pra-gelombang QRIS).
    - ``aug_adapt_idx``: Agt 1 s.d. Agt 10 (awal Agustus matang, 21–31 hari sebelum snapshot).
    - ``aug_val_idx``  : Agt 11 s.d. Agt 21 (evaluasi shift kronologis pasca-adaptasi).
    - ``aug_tail_idx`` : Agt 22 s.d. Agt 31 (ekor tersensor berat, tidak dipakai untuk evaluasi).

    Parameters
    ----------
    df : pd.DataFrame
        Data latih klasifikasi penuh.

    Returns
    -------
    dict[str, np.ndarray]
        Kamus indeks posisi tiap segmen waktu.
    """
    w = pd.to_datetime(df["waktu_dt"] if "waktu_dt" in df.columns else df[TIME_COLUMN])
    return {
        "jan_jul_idx": np.flatnonzero((w <= pd.Timestamp("2025-07-31 23:59:59")).to_numpy()),
        "aug_adapt_idx": np.flatnonzero(
            ((w >= pd.Timestamp("2025-08-01 00:00:00")) & (w <= pd.Timestamp("2025-08-10 23:59:59"))).to_numpy()
        ),
        "aug_val_idx": np.flatnonzero(
            ((w >= pd.Timestamp("2025-08-11 00:00:00")) & (w <= pd.Timestamp("2025-08-21 23:59:59"))).to_numpy()
        ),
        "aug_tail_idx": np.flatnonzero((w > pd.Timestamp("2025-08-21 23:59:59")).to_numpy()),
    }


def _lgbm_val_params(seed: int = SEED) -> dict[str, Any]:
    """Parameter standar LightGBM pembanding antar skema validasi."""
    return {
        "objective": "binary",
        "n_estimators": 350,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }


def _evaluasi_fold(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    fitur: list[str],
    seed: int = SEED,
) -> dict[str, Any]:
    """Latih probe LightGBM dan hitung metrik PR-AUC, ROC-AUC, kalibrasi, serta biaya."""
    cat_in_f = [c for c in CATEGORICAL_COLUMNS if c in fitur]
    tr_al, [va_al] = samakan_tipe_kategori(tr_df, [va_df], cat_cols=cat_in_f)
    clf = lgb.LGBMClassifier(**_lgbm_val_params(seed))
    clf.fit(tr_al[fitur], tr_al[TARGET_COLUMN].to_numpy(dtype=np.int8))
    prob = np.clip(clf.predict_proba(va_al[fitur])[:, 1], 1e-7, 1.0 - 1e-7)

    y_va = va_al[TARGET_COLUMN].to_numpy(dtype=np.int8)
    nom_va = va_al["nominal"].to_numpy(dtype=np.float64)
    flagged = (prob * nom_va > FRAUD_REVIEW_COST_IDR).astype(np.int8)

    cost = hitung_biaya_total_fraud(y_va, flagged, nom_va, review_cost=FRAUD_REVIEW_COST_IDR)
    cost_none = hitung_biaya_total_fraud(y_va, np.zeros_like(y_va), nom_va, review_cost=FRAUD_REVIEW_COST_IDR)
    obs_rate = float(np.mean(y_va))
    mean_prob = float(np.mean(prob))

    return {
        "n_train": int(len(tr_df)),
        "n_val": int(len(va_df)),
        "n_val_fraud": int(y_va.sum()),
        "obs_rate_pct": float(obs_rate * 100.0),
        "mean_prob_pct": float(mean_prob * 100.0),
        "ratio_pred_to_obs": float(mean_prob / max(obs_rate, 1e-12)),
        "pr_auc": float(average_precision_score(y_va, prob)),
        "roc_auc": float(roc_auc_score(y_va, prob)),
        "brier": float(brier_score_loss(y_va, prob)),
        "total_cost_idr": float(cost),
        "cost_flag_none_idr": float(cost_none),
        "cost_per_10k_tx_idr": float(cost / len(va_df) * 10_000.0),
        "cost_reduction_vs_none_pct": float((1.0 - cost / max(cost_none, 1.0)) * 100.0),
        "flag_rate_pct": float(flagged.mean() * 100.0),
    }


def evaluasi_time_cv_matang(
    train_df: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Jalankan Expanding-Window Time CV 3 fold pada periode label matang (Mei, Jun, Jul).

    Parameters
    ----------
    train_df : pd.DataFrame
        Data latih klasifikasi dengan kolom indikator kualitas sudah ditambahkan.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Metrik per fold beserta rata-rata & simpangan baku lintas 3 fold matang.
    """
    folds = buat_fold_waktu_matang(train_df)
    fitur = list(CLEAN_WITH_FLAGS_FEATURES)
    per_fold: list[dict[str, Any]] = []
    for f in folds:
        met = _evaluasi_fold(train_df.iloc[f.train_idx], train_df.iloc[f.val_idx], fitur, seed=seed)
        met["fold"] = f.name
        met["train_end"] = f.train_end
        met["val_period"] = f"{f.val_start} s.d. {f.val_end}"
        per_fold.append(met)

    pr_list = [r["pr_auc"] for r in per_fold]
    roc_list = [r["roc_auc"] for r in per_fold]
    cost10k_list = [r["cost_per_10k_tx_idr"] for r in per_fold]
    ratio_list = [r["ratio_pred_to_obs"] for r in per_fold]

    return {
        "skema": "Expanding-Window Time CV Matang (3 Fold: Mei, Jun, Jul)",
        "per_fold": per_fold,
        "mean_pr_auc": float(np.mean(pr_list)),
        "std_pr_auc": float(np.std(pr_list, ddof=1)),
        "mean_roc_auc": float(np.mean(roc_list)),
        "std_roc_auc": float(np.std(roc_list, ddof=1)),
        "mean_cost_per_10k_tx_idr": float(np.mean(cost10k_list)),
        "std_cost_per_10k_tx_idr": float(np.std(cost10k_list, ddof=1)),
        "mean_ratio_pred_to_obs": float(np.mean(ratio_list)),
    }


def evaluasi_random_stratified_cv(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
) -> dict[str, Any]:
    """Jalankan pembanding Stratified Random 5-Fold CV pada periode matang Jan–Jul.

    Menggunakan periode matang yang sama (Jan–Jul) agar perbedaan skor terhadap
    Time CV semata-mata mencerminkan kebocoran waktu/entitas dari pengacakan,
    bukan pencampuran sensor Agustus.

    Parameters
    ----------
    train_df : pd.DataFrame
        Data latih klasifikasi dengan kolom indikator kualitas sudah ditambahkan.
    n_splits : int, default=5
        Jumlah fold acak terstratifikasi.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Metrik per fold serta rata-rata dan simpangan baku Random 5-Fold CV.
    """
    mat = train_df[train_df["waktu_dt"] <= pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")].reset_index(
        drop=True
    )
    fitur = list(CLEAN_WITH_FLAGS_FEATURES)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_mat = mat[TARGET_COLUMN].to_numpy(dtype=np.int8)

    per_fold: list[dict[str, Any]] = []
    for i, (tr_i, va_i) in enumerate(skf.split(mat, y_mat), start=1):
        met = _evaluasi_fold(mat.iloc[tr_i], mat.iloc[va_i], fitur, seed=seed)
        met["fold"] = f"Random Fold {i}"
        per_fold.append(met)

    pr_list = [r["pr_auc"] for r in per_fold]
    roc_list = [r["roc_auc"] for r in per_fold]
    cost10k_list = [r["cost_per_10k_tx_idr"] for r in per_fold]
    ratio_list = [r["ratio_pred_to_obs"] for r in per_fold]

    return {
        "skema": f"Stratified Random {n_splits}-Fold CV (Jan–Jul Matang)",
        "per_fold": per_fold,
        "mean_pr_auc": float(np.mean(pr_list)),
        "std_pr_auc": float(np.std(pr_list, ddof=1)),
        "mean_roc_auc": float(np.mean(roc_list)),
        "std_roc_auc": float(np.std(roc_list, ddof=1)),
        "mean_cost_per_10k_tx_idr": float(np.mean(cost10k_list)),
        "std_cost_per_10k_tx_idr": float(np.std(cost10k_list, ddof=1)),
        "mean_ratio_pred_to_obs": float(np.mean(ratio_list)),
    }


def evaluasi_validasi_agustus_tersensor(
    train_df: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Ukur distorsi jika validasi diletakkan naif pada Agustus tersensor (Agt 1–31 vs Agt 15–31).

    Melatih satu model pada seluruh periode matang Jan–Jul 2025 lalu mengevaluasinya
    pada empat irisan Agustus yang semakin dekat ke snapshot 31 Agt 2025:
    - ``agt_1_7_matang``       : awal Agustus yang labelnya matang (sudah ada gelombang QRIS).
    - ``agt_1_7_non_qris``     : awal Agustus khusus non-QRIS (kontrol murni tanpa shift QRIS).
    - ``agt_penuh_1_31_naif``  : seluruh Agustus 1–31 (validasi naif bulan terakhir).
    - ``agt_akhir_15_31_berat``: paruh kedua Agustus (tersensor berat, 0–16 hari ke snapshot).

    Parameters
    ----------
    train_df : pd.DataFrame
        Data latih klasifikasi penuh dengan indikator kualitas.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Metrik PR-AUC, rasio kalibrasi, dan biaya tampak pada tiap irisan Agustus.
    """
    fitur = list(CLEAN_WITH_FLAGS_FEATURES)
    w = train_df["waktu_dt"]
    tr_jj = train_df[w <= pd.Timestamp("2025-07-31 23:59:59")].copy()

    irisan = {
        "agt_1_7_matang": train_df[(w >= pd.Timestamp("2025-08-01")) & (w <= pd.Timestamp("2025-08-07 23:59:59"))],
        "agt_1_7_non_qris": train_df[
            (w >= pd.Timestamp("2025-08-01"))
            & (w <= pd.Timestamp("2025-08-07 23:59:59"))
            & (train_df["kanal"] != "qris")
        ],
        "agt_penuh_1_31_naif": train_df[w >= pd.Timestamp("2025-08-01")],
        "agt_akhir_15_31_berat": train_df[w >= pd.Timestamp("2025-08-15")],
    }

    tr_al, aligned = samakan_tipe_kategori(tr_jj, list(irisan.values()), cat_cols=CATEGORICAL_COLUMNS)
    clf = lgb.LGBMClassifier(**_lgbm_val_params(seed))
    clf.fit(tr_al[fitur], tr_al[TARGET_COLUMN].to_numpy(dtype=np.int8))

    out: dict[str, Any] = {}
    for nama, sub in zip(irisan.keys(), aligned):
        prob = np.clip(clf.predict_proba(sub[fitur])[:, 1], 1e-7, 1.0 - 1e-7)
        y_va = sub[TARGET_COLUMN].to_numpy(dtype=np.int8)
        nom_va = sub["nominal"].to_numpy(dtype=np.float64)
        flagged = (prob * nom_va > FRAUD_REVIEW_COST_IDR).astype(np.int8)
        cost = hitung_biaya_total_fraud(y_va, flagged, nom_va, review_cost=FRAUD_REVIEW_COST_IDR)
        cost_none = hitung_biaya_total_fraud(y_va, np.zeros_like(y_va), nom_va, review_cost=FRAUD_REVIEW_COST_IDR)
        obs_rate = float(np.mean(y_va))
        mean_prob = float(np.mean(prob))
        out[nama] = {
            "n_val": int(len(sub)),
            "n_val_fraud": int(y_va.sum()),
            "obs_rate_pct": float(obs_rate * 100.0),
            "mean_prob_pct": float(mean_prob * 100.0),
            "ratio_pred_to_obs": float(mean_prob / max(obs_rate, 1e-12)),
            "pr_auc": float(average_precision_score(y_va, prob)),
            "roc_auc": float(roc_auc_score(y_va, prob)),
            "brier": float(brier_score_loss(y_va, prob)),
            "total_cost_idr": float(cost),
            "cost_flag_none_idr": float(cost_none),
            "cost_per_10k_tx_idr": float(cost / len(sub) * 10_000.0),
            "flag_rate_pct": float(flagged.mean() * 100.0),
        }
    return out


def simpan_plot_perbandingan_validasi(
    hasil: dict[str, Any],
    output_path: Path = FIGURES_DIR / "task2_2b_validation_comparison.png",
) -> Path:
    """Visualisasikan perbandingan Time CV Matang vs Random 5-Fold vs Validasi Agustus Tersensor.

    Parameters
    ----------
    hasil : dict[str, Any]
        Kamus hasil perbandingan validasi Tugas 2b.
    output_path : Path
        Lokasi penyimpanan gambar PNG beresolusi 300 DPI.

    Returns
    -------
    Path
        Path gambar tersimpan.
    """
    atur_gaya_grafik()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.75))

    tc = hasil["time_cv_matang"]
    rc = hasil["random_cv_matang"]
    ac = hasil["validasi_agustus_tersensor"]

    # Panel A: Perbandingan PR-AUC antar skema validasi
    ax0 = axes[0]
    labels_a = [
        "Random\n5-Fold",
        "Time CV\n(Mei–Jul)",
        "Agt 1–7\n(non-QRIS)",
        "Agt 1–31\n(naif)",
        "Agt 15–31\n(tersensor)",
    ]
    vals_a = [
        rc["mean_pr_auc"],
        tc["mean_pr_auc"],
        ac["agt_1_7_non_qris"]["pr_auc"],
        ac["agt_penuh_1_31_naif"]["pr_auc"],
        ac["agt_akhir_15_31_berat"]["pr_auc"],
    ]
    errs_a = [rc["std_pr_auc"], tc["std_pr_auc"], 0.0, 0.0, 0.0]
    cols_a = ["#9467bd", "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    x0 = np.arange(len(labels_a))
    bars0 = ax0.bar(x0, vals_a, color=cols_a, width=0.58)
    ax0.errorbar(x0[:2], vals_a[:2], yerr=errs_a[:2], fmt="none", ecolor="#222222", elinewidth=1.0, capsize=2.5)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(labels_a, fontsize=6.2)
    ax0.set_ylim(0, max(vals_a) * 1.22)
    ax0.set_ylabel("PR-AUC")
    ax0.set_title("(a) Distorsi PR-AUC antar skema validasi")
    sumbu_koma(ax0, "y")
    for bar, val, err in zip(bars0, vals_a, errs_a):
        ax0.annotate(
            angka_id(val, 3),
            (bar.get_x() + bar.get_width() / 2, val + err),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )

    # Panel B: Rasio kalibrasi semu (mean_prob / obs_rate) pada tiap jendela
    ax1 = axes[1]
    labels_b = [
        "Fold 1\n(Mei)",
        "Fold 2\n(Jun)",
        "Fold 3\n(Jul)",
        "Agt 1–31\n(naif)",
        "Agt 15–31\n(tersensor)",
    ]
    ratios_b = [
        tc["per_fold"][0]["ratio_pred_to_obs"],
        tc["per_fold"][1]["ratio_pred_to_obs"],
        tc["per_fold"][2]["ratio_pred_to_obs"],
        ac["agt_penuh_1_31_naif"]["ratio_pred_to_obs"],
        ac["agt_akhir_15_31_berat"]["ratio_pred_to_obs"],
    ]
    cols_b = ["#2ca02c", "#2ca02c", "#2ca02c", "#ff7f0e", "#d62728"]
    x1 = np.arange(len(labels_b))
    bars1 = ax1.bar(x1, ratios_b, color=cols_b, width=0.58)
    ax1.axhline(1.0, color="#222222", linestyle="--", linewidth=0.9, label="Ideal (1,00)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels_b, fontsize=6.2)
    ax1.set_ylim(0, max(ratios_b) * 1.22)
    ax1.set_ylabel("Rasio rerata prediksi / laju teramati")
    ax1.set_title(f"(b) {miring('Over')}-{miring('prediction')} semu akibat {miring('right')}-{miring('censoring')}")
    sumbu_koma(ax1, "y")
    label_batang(ax1, bars1, desimal=2, ukuran=5.8)
    ax1.legend(loc="upper left", fontsize=6.0)

    fig.tight_layout(w_pad=1.5)
    siapkan_direktori()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def jalankan_validasi_klasifikasi(
    mode: str = "all",
    simpan_json: bool = True,
) -> dict[str, Any]:
    """Jalankan eksperimen perbandingan validasi Tugas 2b sesuai mode pilihan.

    Parameters
    ----------
    mode : {'time', 'random', 'censored', 'all'}, default='all'
        Skema validasi yang dieksekusi.
    simpan_json : bool, default=True
        Bila ``True`` (dan ``mode == 'all'``), simpan hasil ke ``task2_validation.json``.

    Returns
    -------
    dict[str, Any]
        Ringkasan hasil validasi Tugas 2b.
    """
    atur_seed_global(SEED)
    siapkan_direktori()

    with catat_waktu("Muat & siapkan data latih klasifikasi"):
        train_raw = pd.read_csv(KLASIFIKASI_TRAIN_PATH)
        train_raw["waktu_dt"] = pd.to_datetime(train_raw[TIME_COLUMN])
        train_df = tambah_flag_kualitas_data(train_raw)

    hasil: dict[str, Any] = {
        "aturan_anti_censoring": {
            "snapshot_date": str(SNAPSHOT_TIMESTAMP.date()),
            "mature_cutoff_date": MATURE_TRAIN_END_DATE,
            "min_maturity_lag_days": MIN_MATURITY_LAG_DAYS,
            "alasan": "Label is_fraud ditarik pada 2025-08-31 dengan jeda konfirmasi ~21-25 hari "
            "(29-31 Agt bernilai 0 fraud). Validasi utama hanya memakai 3 fold matang (Mei, Jun, Jul) "
            "agar metrik PR-AUC, kalibrasi, dan biaya tidak terdistorsi label belum lengkap.",
        }
    }

    if mode in ("time", "all"):
        with catat_waktu("2b.1 Expanding-Window Time CV Matang (3 Fold: Mei, Jun, Jul)"):
            hasil["time_cv_matang"] = evaluasi_time_cv_matang(train_df, seed=SEED)
            tc = hasil["time_cv_matang"]
            print(
                f"  Time CV Matang: PR-AUC={tc['mean_pr_auc']:.4f} ± {tc['std_pr_auc']:.4f}, "
                f"Biaya/10k={tc['mean_cost_per_10k_tx_idr']:,.0f} Rp, rasio={tc['mean_ratio_pred_to_obs']:.2f}"
            )

    if mode in ("random", "all"):
        with catat_waktu("2b.2 Pembanding Stratified Random 5-Fold CV (Jan–Jul Matang)"):
            hasil["random_cv_matang"] = evaluasi_random_stratified_cv(train_df, n_splits=5, seed=SEED)
            rc = hasil["random_cv_matang"]
            print(
                f"  Random 5-Fold : PR-AUC={rc['mean_pr_auc']:.4f} ± {rc['std_pr_auc']:.4f}, "
                f"Biaya/10k={rc['mean_cost_per_10k_tx_idr']:,.0f} Rp"
            )

    if mode in ("censored", "all"):
        with catat_waktu("2b.3 Evaluasi Naif pada Agustus Tersensor (Agt 1–31 & 15–31)"):
            hasil["validasi_agustus_tersensor"] = evaluasi_validasi_agustus_tersensor(train_df, seed=SEED)
            ac = hasil["validasi_agustus_tersensor"]
            print(
                f"  Agt 1–7 non-QRIS: PR-AUC={ac['agt_1_7_non_qris']['pr_auc']:.4f}, rasio={ac['agt_1_7_non_qris']['ratio_pred_to_obs']:.2f}"
            )
            print(
                f"  Agt 1–31 naif   : PR-AUC={ac['agt_penuh_1_31_naif']['pr_auc']:.4f}, rasio={ac['agt_penuh_1_31_naif']['ratio_pred_to_obs']:.2f}"
            )
            print(
                f"  Agt 15–31 berat : PR-AUC={ac['agt_akhir_15_31_berat']['pr_auc']:.4f}, rasio={ac['agt_akhir_15_31_berat']['ratio_pred_to_obs']:.2f}"
            )

    if mode == "all":
        tc = hasil["time_cv_matang"]
        rc = hasil["random_cv_matang"]
        ac = hasil["validasi_agustus_tersensor"]
        hasil["ringkasan_selisih"] = {
            "optimisme_random_vs_time_pr_auc": float(rc["mean_pr_auc"] - tc["mean_pr_auc"]),
            "optimisme_random_vs_time_pct": float((rc["mean_pr_auc"] / tc["mean_pr_auc"] - 1.0) * 100.0),
            "rasio_std_time_vs_random_pr_auc": float(tc["std_pr_auc"] / max(rc["std_pr_auc"], 1e-12)),
            "penurunan_pr_auc_naif_agt_vs_time": float(
                ac["agt_penuh_1_31_naif"]["pr_auc"] - tc["mean_pr_auc"]
            ),
            "penurunan_pr_auc_agt15_31_vs_time": float(
                ac["agt_akhir_15_31_berat"]["pr_auc"] - tc["mean_pr_auc"]
            ),
            "rasio_kalibrasi_agt15_31": float(ac["agt_akhir_15_31_berat"]["ratio_pred_to_obs"]),
        }
        simpan_plot_perbandingan_validasi(hasil)
        if simpan_json:
            out_path = RESULTS_DIR / "task2_validation.json"
            tulis_json(hasil, out_path)
            print(f"[SAVED] {out_path.name}")

    return hasil


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2b anti-censoring time-based validation.")
    parser.add_argument(
        "--mode",
        choices=["time", "random", "censored", "all"],
        default="all",
        help="Validation mode to run (default: all).",
    )
    args = parser.parse_args()
    jalankan_validasi_klasifikasi(mode=args.mode, simpan_json=(args.mode == "all"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
