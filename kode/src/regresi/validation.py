"""Desain Validasi Berbasis Waktu & Perbandingan vs Random K-Fold (Tugas 1b).

Modul ini menyediakan skema validasi resmi Tugas 1 (Regresi) yang meniru
pemakaian nyata (memprediksi 2 bulan ke depan dari bulan-bulan sebelumnya)
serta membandingkannya secara objektif dengan Random 5-Fold CV menggunakan
model dan fitur bersih yang identik.

Jalankan langsung:
    python kode/src/regresi/validation.py                 # Jalankan time vs random + plot + JSON
    python kode/src/regresi/validation.py --mode time     # Hanya Expanding-Window Time CV
    python kode/src/regresi/validation.py --mode random   # Hanya Random 5-Fold CV
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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    N_JOBS,
    REGRESI_TRAIN_PATH,
    RESULTS_DIR,
    SEED,
    siapkan_direktori,
)
from src.regresi.preprocess import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    DATE_COLUMN,
    NAMED_NUMERIC_COLUMNS,
    SIGNAL_F_COLUMNS,
    TARGET_COLUMN,
    BeratImputer,
    RouteDistanceImputer,
    align_categorical_dtypes,
    fix_target_unit_errors,
)
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    atur_gaya_grafik,
    label_batang,
    miring,
    hitung_cakupan_p90,
    hitung_pinball_loss,
    tulis_json,
    atur_seed_global,
    sumbu_koma,
    catat_waktu,
)

# Definisi 3 fold Expanding-Window berhorizon 2 bulan (meniru test Sep–Okt 2025)
TIME_FOLD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("Fold 1 (Mar–Apr)", "2025-02-28", "2025-03-01", "2025-04-30"),
    ("Fold 2 (Mei–Jun)", "2025-04-30", "2025-05-01", "2025-06-30"),
    ("Fold 3 (Jul–Agu)", "2025-06-30", "2025-07-01", "2025-08-31"),
)

CLEAN_FEATURE_COLUMNS: tuple[str, ...] = (
    CATEGORICAL_COLUMNS + NAMED_NUMERIC_COLUMNS + SIGNAL_F_COLUMNS
)


@dataclass(frozen=True)
class TimeFold:
    """Satu fold validasi berbasis waktu dengan batas tanggal eksplisit.

    Attributes
    ----------
    name : str
        Label ringkas fold (mis. ``'Fold 3 (Jul–Agu)'``).
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


def buat_fold_waktu(df: pd.DataFrame) -> list[TimeFold]:
    """Bangun 3 fold Expanding-Window berhorizon 2 bulan tanpa kebocoran masa depan.

    Setiap fold meniru persis tugas produksi (memprediksi 2 bulan penuh ke depan):
    - Fold 1: Train Jan–Feb 2025 -> Val Mar–Apr 2025 (uji guncangan Lebaran).
    - Fold 2: Train Jan–Apr 2025 -> Val Mei–Jun 2025 (uji periode pasca-Lebaran).
    - Fold 3: Train Jan–Jun 2025 -> Val Jul–Agu 2025 (uji shift K07 & antrean naik).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame latih yang memuat kolom ``tgl_pickup``.

    Returns
    -------
    list[TimeFold]
        Daftar 3 objek :class:`TimeFold` terurut secara kronologis.
    """
    dt = pd.to_datetime(df[DATE_COLUMN])
    folds: list[TimeFold] = []
    for name, tr_end, va_start, va_end in TIME_FOLD_SPECS:
        tr_idx = np.flatnonzero(dt <= tr_end)
        va_idx = np.flatnonzero((dt >= va_start) & (dt <= va_end))
        folds.append(
            TimeFold(
                name=name,
                train_end=tr_end,
                val_start=va_start,
                val_end=va_end,
                train_idx=tr_idx,
                val_idx=va_idx,
            )
        )
    return folds


def _evaluasi_satu_fold(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Latih & evaluasi probe LightGBM Quantile pada satu pasangan train–val anti-leakage.

    Seluruh transformasi (``RouteDistanceImputer``, ``BeratImputer``,
    ``align_categorical_dtypes``, dan koreksi target ``/ 60``) di-``fit`` ketat
    hanya pada ``tr_df`` di dalam fold ini.

    Parameters
    ----------
    tr_df : pd.DataFrame
        Data latih fold aktif.
    va_df : pd.DataFrame
        Data validasi fold aktif.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Jumlah baris train/val serta metrik Pinball P50, Pinball P90, P90 Coverage, dan MAE.
    """
    y_tr, _ = fix_target_unit_errors(tr_df[TARGET_COLUMN])
    y_va, _ = fix_target_unit_errors(va_df[TARGET_COLUMN])

    dist_imp = RouteDistanceImputer().fit(tr_df)
    berat_imp = BeratImputer().fit(tr_df)
    tr_clean = berat_imp.transform(dist_imp.transform(tr_df, add_indicator=True), add_indicator=True)
    va_clean = berat_imp.transform(dist_imp.transform(va_df, add_indicator=True), add_indicator=True)

    feat_cols = list(CLEAN_FEATURE_COLUMNS) + ["jarak_km_was_corrupt", "berat_kg_missing"]
    X_tr, [X_va] = align_categorical_dtypes(tr_clean[feat_cols], [va_clean[feat_cols]])

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }
    m50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **params).fit(X_tr, y_tr)
    m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **params).fit(X_tr, y_tr)

    p50 = m50.predict(X_va)
    p90 = np.maximum(p50, m90.predict(X_va))

    return {
        "n_train": int(len(tr_df)),
        "n_val": int(len(va_df)),
        "pinball_p50": round(hitung_pinball_loss(y_va, p50, 0.5), 4),
        "pinball_p90": round(hitung_pinball_loss(y_va, p90, 0.9), 4),
        "p90_coverage": round(hitung_cakupan_p90(y_va, p90), 4),
        "mae_p50": round(float(mean_absolute_error(y_va, p50)), 4),
    }


def _ringkas_fold(fold_results: list[dict[str, Any]]) -> dict[str, float]:
    """Hitung rata-rata dan simpangan baku metrik lintas fold."""
    summary: dict[str, float] = {}
    for key in ("pinball_p50", "pinball_p90", "p90_coverage", "mae_p50"):
        vals = [f[key] for f in fold_results]
        summary[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"{key}_std"] = round(float(np.std(vals, ddof=0)), 4)
    return summary


def evaluasi_validasi_waktu(train: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """Jalankan Expanding-Window Time-Based CV (3 fold x 2 bulan) pada data latih.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Hasil metrik per fold waktu beserta rata-rata dan simpangan bakunya.
    """
    folds = buat_fold_waktu(train)
    per_fold: list[dict[str, Any]] = []
    for f in folds:
        tr_df = train.iloc[f.train_idx].reset_index(drop=True)
        va_df = train.iloc[f.val_idx].reset_index(drop=True)
        res = _evaluasi_satu_fold(tr_df, va_df, seed=seed)
        res["fold"] = f.name
        res["periode_train"] = f"2025-01-01..{f.train_end}"
        res["periode_val"] = f"{f.val_start}..{f.val_end}"
        per_fold.append(res)

    return {
        "skema": "Expanding-Window Forward CV (3 fold x 2 bulan)",
        "per_fold": per_fold,
        "ringkasan": _ringkas_fold(per_fold),
    }


def evaluasi_random_kfold(
    train: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
) -> dict[str, Any]:
    """Jalankan Random K-Fold CV sebagai pembanding untuk mengukur bias optimisme.

    Menggunakan model, fitur bersih, dan preprocessing anti-leakage per fold
    yang persis sama dengan :func:`evaluasi_validasi_waktu`, sehingga perbedaan
    skor murni berasal dari cara pemisahan data (acak vs kronologis).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.
    n_splits : int, default=5
        Jumlah fold acak.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Hasil metrik per fold acak beserta rata-rata dan simpangan bakunya.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_fold: list[dict[str, Any]] = []
    for idx, (tr_idx, va_idx) in enumerate(kf.split(train), start=1):
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)
        res = _evaluasi_satu_fold(tr_df, va_df, seed=seed)
        res["fold"] = f"Random Fold {idx}"
        per_fold.append(res)

    return {
        "skema": f"Random {n_splits}-Fold CV (shuffle=True)",
        "per_fold": per_fold,
        "ringkasan": _ringkas_fold(per_fold),
    }


def plot_perbandingan_validasi(
    hasil_waktu: dict[str, Any],
    hasil_random: dict[str, Any],
) -> Path:
    """Buat grafik 2-panel perbandingan Time-Based CV vs Random 5-Fold CV.

    Panel (a): Pinball P50 & P90 untuk tiap fold waktu, rata-rata Time CV,
    dan rata-rata Random 5-Fold (memperlihatkan kesenjangan optimisme Random CV).
    Panel (b): P90 Coverage per fold waktu vs Random 5-Fold terhadap target 90%.

    Parameters
    ----------
    hasil_waktu : dict[str, Any]
        Keluaran dari :func:`evaluasi_validasi_waktu`.
    hasil_random : dict[str, Any]
        Keluaran dari :func:`evaluasi_random_kfold`.

    Returns
    -------
    Path
        Path file PNG yang tersimpan di ``kode/figures/task1_1b_validation_comparison.png``.
    """
    siapkan_direktori()
    atur_gaya_grafik()

    # Periode tiap fold tercantum pada tabel laporan, sehingga label cukup memuat nomornya.
    n_fold = len(hasil_waktu["per_fold"])
    labels = [f"{miring('Fold')} {i}" for i in range(1, n_fold + 1)] + [
        "Rata-rata\nberbasis waktu",
        "Rata-rata\nacak",
    ]

    p50_vals = [f["pinball_p50"] for f in hasil_waktu["per_fold"]] + [
        hasil_waktu["ringkasan"]["pinball_p50_mean"],
        hasil_random["ringkasan"]["pinball_p50_mean"],
    ]
    p90_vals = [f["pinball_p90"] for f in hasil_waktu["per_fold"]] + [
        hasil_waktu["ringkasan"]["pinball_p90_mean"],
        hasil_random["ringkasan"]["pinball_p90_mean"],
    ]
    cov_vals = [f["p90_coverage"] * 100.0 for f in hasil_waktu["per_fold"]] + [
        hasil_waktu["ringkasan"]["p90_coverage_mean"] * 100.0,
        hasil_random["ringkasan"]["p90_coverage_mean"] * 100.0,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.6))
    x = np.arange(len(labels))
    w = 0.38
    batas_acak = n_fold + 0.5  # pemisah antara validasi berbasis waktu dan validasi acak

    # Panel (a): Pinball loss P50 dan P90
    b1 = axes[0].bar(x - w / 2, p50_vals, width=w, color="#2563eb", label="P50")
    b2 = axes[0].bar(x + w / 2, p90_vals, width=w, color="#f97316", label="P90")
    label_batang(axes[0], b1, putar=True)
    label_batang(axes[0], b2, putar=True)
    axes[0].set_ylabel(f"{miring('Pinball loss')} (jam)")
    axes[0].set_title(f"(a) {miring('Pinball Loss')} per Skema Validasi")
    axes[0].set_ylim(0, max(p50_vals + p90_vals) * 1.28)
    axes[0].legend(loc="upper right", ncol=2)

    # Panel (b): Cakupan P90 terhadap sasaran 90%
    warna = ["#93c5fd"] * n_fold + ["#2563eb", "#f97316"]
    bc = axes[1].bar(x, cov_vals, width=0.58, color=warna)
    axes[1].axhline(90.0, color="#111827", linestyle="--", linewidth=0.9, label="Sasaran 90%")
    label_batang(axes[1], bc, desimal=1)
    axes[1].set_ylabel("Cakupan P90 (%)")
    axes[1].set_title("(b) Cakupan P90 per Skema Validasi")
    axes[1].set_ylim(75, 96)
    axes[1].set_yticks(np.arange(75, 96, 5))
    axes[1].legend(loc="upper left")

    for ax in axes:
        ax.axvline(batas_acak, color="#6b7280", linestyle=":", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)
    sumbu_koma(axes[0], "y")

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1b_validation_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _cetak_ringkasan(hasil_waktu: dict[str, Any] | None, hasil_random: dict[str, Any] | None) -> None:
    """Cetak tabel perbandingan validasi waktu vs random ke terminal."""
    if hasil_waktu is not None:
        print("=== [time] EXPANDING-WINDOW TIME-BASED CV (3 FOLD x 2 BULAN) ===")
        for f in hasil_waktu["per_fold"]:
            print(
                f"  {f['fold']:18s} (tr={f['n_train']:>7,}, va={f['n_val']:>6,}) -> "
                f"P50={f['pinball_p50']:.4f} | P90={f['pinball_p90']:.4f} | "
                f"cov90={f['p90_coverage']*100:.2f}% | MAE={f['mae_p50']:.2f}"
            )
        rw = hasil_waktu["ringkasan"]
        print(
            f"  Rata-rata Time CV -> "
            f"P50={rw['pinball_p50_mean']:.4f} (±{rw['pinball_p50_std']:.4f}) | "
            f"P90={rw['pinball_p90_mean']:.4f} (±{rw['pinball_p90_std']:.4f}) | "
            f"cov90={rw['p90_coverage_mean']*100:.2f}%"
        )

    if hasil_random is not None:
        print("\n=== [random] RANDOM 5-FOLD CV (SHUFFLE=TRUE) ===")
        for f in hasil_random["per_fold"]:
            print(
                f"  {f['fold']:18s} (tr={f['n_train']:>7,}, va={f['n_val']:>6,}) -> "
                f"P50={f['pinball_p50']:.4f} | P90={f['pinball_p90']:.4f} | "
                f"cov90={f['p90_coverage']*100:.2f}% | MAE={f['mae_p50']:.2f}"
            )
        rr = hasil_random["ringkasan"]
        print(
            f"  Rata-rata Random  -> "
            f"P50={rr['pinball_p50_mean']:.4f} (±{rr['pinball_p50_std']:.4f}) | "
            f"P90={rr['pinball_p90_mean']:.4f} (±{rr['pinball_p90_std']:.4f}) | "
            f"cov90={rr['p90_coverage_mean']*100:.2f}%"
        )

    if hasil_waktu is not None and hasil_random is not None:
        rw = hasil_waktu["ringkasan"]
        rr = hasil_random["ringkasan"]
        gap_p50 = (rw["pinball_p50_mean"] - rr["pinball_p50_mean"]) / rr["pinball_p50_mean"] * 100.0
        gap_p90 = (rw["pinball_p90_mean"] - rr["pinball_p90_mean"]) / rr["pinball_p90_mean"] * 100.0
        print("\n=== PERBANDINGAN OPTIMISM GAP (TIME CV vs RANDOM 5-FOLD) ===")
        print(f"  Pinball P50 : Time CV {rw['pinball_p50_mean']:.4f} vs Random {rr['pinball_p50_mean']:.4f} (Time {gap_p50:+.1f}% lebih tinggi)")
        print(f"  Pinball P90 : Time CV {rw['pinball_p90_mean']:.4f} vs Random {rr['pinball_p90_mean']:.4f} (Time {gap_p90:+.1f}% lebih tinggi)")
        print(
            f"  P90 Coverage: Time CV {rw['p90_coverage_mean']*100:.2f}% "
            f"(Fold 3 Jul–Agu: {hasil_waktu['per_fold'][2]['p90_coverage']*100:.2f}%) "
            f"vs Random {rr['p90_coverage_mean']*100:.2f}%"
        )


def main() -> None:
    """Jalankan benchmark validasi Tugas 1b dari CLI."""
    atur_seed_global(SEED)
    siapkan_direktori()

    parser = argparse.ArgumentParser(description="Task 1b validation benchmark (Time CV vs Random K-Fold).")
    parser.add_argument(
        "--mode",
        choices=("time", "random", "all"),
        default="all",
        help="Validation scheme to run (default: all).",
    )
    args = parser.parse_args()

    train = pd.read_csv(REGRESI_TRAIN_PATH)

    hasil_waktu: dict[str, Any] | None = None
    hasil_random: dict[str, Any] | None = None

    if args.mode in ("time", "all"):
        with catat_waktu("Evaluasi Expanding-Window Time-Based CV (3 fold)"):
            hasil_waktu = evaluasi_validasi_waktu(train, seed=SEED)

    if args.mode in ("random", "all"):
        with catat_waktu("Evaluasi Random 5-Fold CV"):
            hasil_random = evaluasi_random_kfold(train, n_splits=5, seed=SEED)

    _cetak_ringkasan(hasil_waktu, hasil_random)

    if hasil_waktu is not None and hasil_random is not None:
        fig_path = plot_perbandingan_validasi(hasil_waktu, hasil_random)
        out_json = RESULTS_DIR / "task1_validation.json"
        tulis_json(
            {
                "task": "Tugas 1b — Desain Validasi Berbasis Waktu vs Random K-Fold",
                "seed": SEED,
                "time_based_cv": hasil_waktu,
                "random_kfold_cv": hasil_random,
                "figure": fig_path.name,
            },
            out_json,
        )
        print(f"\nGrafik tersimpan di : {fig_path.relative_to(KODE_DIR.parent)}")
        print(f"Hasil JSON tersimpan: {out_json.relative_to(KODE_DIR.parent)}")


if __name__ == "__main__":
    main()
