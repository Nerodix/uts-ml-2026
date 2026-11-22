"""Pengukuran dampak kuantitatif setiap temuan audit Tugas 1a (Langkah 1a.6b).

Setiap temuan audit diukur dampaknya secara terpisah pada split kronologis
jujur (Train Jan–Jun 2025 -> Validasi Jul–Agu 2025) dengan pola satu temuan
= satu fungsi eksperimen:
1. ``ukur_dampak_leakage_scan``  : Leakage ``jumlah_scan`` (CV naif vs simulasi test vs jujur).
2. ``ukur_dampak_satuan_target`` : Error satuan target > 1000 jam (Ridge & LightGBM Quantile).
3. ``ukur_dampak_pemulihan_jarak``: Sentinel ``jarak_km`` (0 & 9999) pada subset sentinel & global.
4. ``ukur_dampak_imputasi_berat`` : Imputasi ``berat_kg`` (median global vs volume * densitas).
5. ``ukur_dampak_seleksi_f``      : Memangkas 40 fitur noise ``f`` (termasuk drift ``f30``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FIGURES_DIR, N_JOBS, SEED, ensure_dirs
from src.regresi.preprocess import (
    ANONYMOUS_F_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMN,
    LEAKAGE_COLUMNS,
    NAMED_NUMERIC_COLUMNS,
    SIGNAL_F_COLUMNS,
    TARGET_COLUMN,
    BeratImputer,
    RouteDistanceImputer,
    align_categorical_dtypes,
    fix_target_unit_errors,
    identify_corrupt_jarak_mask,
)
from src.utils import configure_plotting, p90_coverage, pinball_loss

AUDIT_VAL_CUTOFF = "2025-07-01"


@dataclass
class SplitDampak:
    """Wadah split kronologis Jan–Jun -> Jul–Agu beserta fitur bersih anti-leakage."""

    tr_raw: pd.DataFrame
    va_raw: pd.DataFrame
    tr_clean: pd.DataFrame
    va_clean: pd.DataFrame
    y_tr_raw: np.ndarray
    y_tr_clean: np.ndarray
    y_va_clean: np.ndarray
    berat_imputer: BeratImputer
    honest_metrics: dict[str, float] | None = None
    honest_pred_p50: np.ndarray | None = None
    honest_pred_p90: np.ndarray | None = None


def _lgbm_params(seed: int = SEED) -> dict[str, Any]:
    """Parameter standar probe LightGBM Quantile agar seluruh eksperimen setara."""
    return {
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


def _fit_predict_lgbm_quantile(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Latih pasangan LightGBM Quantile (P50 & P90) dan kembalikan prediksi terurut."""
    params = _lgbm_params(seed)
    m50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **params).fit(X_train, y_train)
    m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **params).fit(X_train, y_train)
    p50 = m50.predict(X_val)
    p90 = np.maximum(p50, m90.predict(X_val))
    return p50, p90


def _hitung_metrik_kuantil(
    y_true: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> dict[str, float]:
    """Hitung metrik evaluasi resmi Tugas 1 (Pinball P50, Pinball P90, Coverage, MAE)."""
    return {
        "pinball_p50": round(pinball_loss(y_true, p50, 0.5), 4),
        "pinball_p90": round(pinball_loss(y_true, p90, 0.9), 4),
        "p90_coverage": round(p90_coverage(y_true, p90), 4),
        "mae_p50": round(float(mean_absolute_error(y_true, p50)), 4),
    }


def siapkan_split_dampak(train: pd.DataFrame) -> SplitDampak:
    """Buat split kronologis (Jan–Jun -> Jul–Agu) dan latih model acuan jujur sekali saja.

    Catatan Anti-Leakage
    --------------------
    ``RouteDistanceImputer`` dan ``BeratImputer`` di-``fit`` hanya pada data
    Jan–Jun 2025, lalu di-``transform`` ke Jul–Agu 2025. Evaluasi selalu memakai
    target validasi bersih (``y_va_clean``) agar perbandingan antar eksperimen adil.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih lengkap ``regresi_train.csv.gz``.

    Returns
    -------
    SplitDampak
        Objek split siap pakai untuk seluruh 5 fungsi pengukuran dampak.
    """
    dt = pd.to_datetime(train[DATE_COLUMN])
    tr_raw = train.loc[dt < AUDIT_VAL_CUTOFF].reset_index(drop=True)
    va_raw = train.loc[dt >= AUDIT_VAL_CUTOFF].reset_index(drop=True)

    y_tr_raw = tr_raw[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_tr_clean, _ = fix_target_unit_errors(y_tr_raw)
    y_va_clean, _ = fix_target_unit_errors(va_raw[TARGET_COLUMN])

    dist_imp = RouteDistanceImputer().fit(tr_raw)
    berat_imp = BeratImputer().fit(tr_raw)
    tr_clean = berat_imp.transform(dist_imp.transform(tr_raw, add_indicator=False), add_indicator=False)
    va_clean = berat_imp.transform(dist_imp.transform(va_raw, add_indicator=False), add_indicator=False)

    # Latih acuan jujur (tanpa jumlah_scan, target /60, jarak pulih, semua 50 f) sekali saja
    base_cols = list(CATEGORICAL_COLUMNS) + list(NAMED_NUMERIC_COLUMNS) + list(ANONYMOUS_F_COLUMNS)
    X_tr, [X_va] = align_categorical_dtypes(tr_clean[base_cols], [va_clean[base_cols]])
    p50_honest, p90_honest = _fit_predict_lgbm_quantile(X_tr, y_tr_clean, X_va)

    return SplitDampak(
        tr_raw=tr_raw,
        va_raw=va_raw,
        tr_clean=tr_clean,
        va_clean=va_clean,
        y_tr_raw=y_tr_raw,
        y_tr_clean=y_tr_clean,
        y_va_clean=y_va_clean,
        berat_imputer=berat_imp,
        honest_metrics=_hitung_metrik_kuantil(y_va_clean, p50_honest, p90_honest),
        honest_pred_p50=p50_honest,
        honest_pred_p90=p90_honest,
    )


def ukur_dampak_leakage_scan(split: SplitDampak, test: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Ukur dampak leakage ``jumlah_scan`` pada CV naif vs simulasi test vs model jujur.

    Model yang sama (dilatih dengan ``jumlah_scan`` pasca-kirim) dievaluasi dalam
    dua kondisi: memakai ``jumlah_scan`` validasi mentah (menipu) dan memakai
    ``jumlah_scan`` yang disampel dari distribusi pickup-time data uji (kondisi nyata).

    Parameters
    ----------
    split : SplitDampak
        Objek split kronologis dari :func:`siapkan_split_dampak`.
    test : pd.DataFrame
        Data uji regresi (diambil distribusi ``jumlah_scan`` untuk simulasi produksi).

    Returns
    -------
    dict[str, dict[str, float]]
        Metrik P50, P90, coverage, dan MAE untuk ketiga skenario.
    """
    leaky_cols = (
        list(CATEGORICAL_COLUMNS)
        + list(NAMED_NUMERIC_COLUMNS)
        + list(ANONYMOUS_F_COLUMNS)
        + list(LEAKAGE_COLUMNS)
    )
    X_tr, [X_va_naive] = align_categorical_dtypes(split.tr_clean[leaky_cols], [split.va_clean[leaky_cols]])

    params = _lgbm_params(SEED)
    m50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **params).fit(X_tr, split.y_tr_clean)
    m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **params).fit(X_tr, split.y_tr_clean)

    p50_naive = m50.predict(X_va_naive)
    p90_naive = np.maximum(p50_naive, m90.predict(X_va_naive))

    # Simulasi produksi: ganti jumlah_scan validasi ke sebaran pickup-time di test
    rng = np.random.default_rng(SEED)
    X_va_deploy = X_va_naive.copy()
    X_va_deploy["jumlah_scan"] = rng.choice(
        test["jumlah_scan"].to_numpy(), size=len(X_va_deploy), replace=True
    )
    p50_deploy = m50.predict(X_va_deploy)
    p90_deploy = np.maximum(p50_deploy, m90.predict(X_va_deploy))

    assert split.honest_metrics is not None
    return {
        "dengan_scan_cv_naif": _hitung_metrik_kuantil(split.y_va_clean, p50_naive, p90_naive),
        "dengan_scan_simulasi_test": _hitung_metrik_kuantil(split.y_va_clean, p50_deploy, p90_deploy),
        "tanpa_scan_jujur": split.honest_metrics,
    }


def ukur_dampak_satuan_target(split: SplitDampak) -> dict[str, dict[str, float]]:
    """Ukur dampak error satuan target (> 1000 jam = menit) pada Ridge dan LightGBM Quantile.

    Memperlihatkan kontras penting: model linear L2 (Ridge) mengalami bias besar
    karena gradien MSE meledak terhadap 370 outlier ``* 60``, sedangkan LightGBM
    Quantile relatif kebal karena gradien pinball loss terbatas.

    Parameters
    ----------
    split : SplitDampak
        Objek split kronologis dari :func:`siapkan_split_dampak`.

    Returns
    -------
    dict[str, dict[str, float]]
        Perbandingan metrik target mentah vs target ``/ 60`` pada Ridge dan LightGBM.
    """
    # 1. LightGBM Quantile pada target mentah vs target bersih
    base_cols = list(CATEGORICAL_COLUMNS) + list(NAMED_NUMERIC_COLUMNS) + list(ANONYMOUS_F_COLUMNS)
    X_tr, [X_va] = align_categorical_dtypes(split.tr_clean[base_cols], [split.va_clean[base_cols]])
    p50_raw, p90_raw = _fit_predict_lgbm_quantile(X_tr, split.y_tr_raw, X_va)

    # 2. Ridge Regression (L2) pada target mentah vs target bersih
    ridge_cats = ["layanan", "mitra_kurir", "hub_asal", "pulau_tujuan"]
    ridge_nums = list(NAMED_NUMERIC_COLUMNS) + list(SIGNAL_F_COLUMNS)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ridge_nums),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ridge_cats),
        ]
    )
    ridge_raw = Pipeline([("pre", pre), ("reg", Ridge(alpha=10.0, random_state=SEED))])
    ridge_clean = Pipeline([("pre", pre), ("reg", Ridge(alpha=10.0, random_state=SEED))])
    ridge_raw.fit(split.tr_clean[ridge_nums + ridge_cats], split.y_tr_raw)
    ridge_clean.fit(split.tr_clean[ridge_nums + ridge_cats], split.y_tr_clean)

    pr_raw = ridge_raw.predict(split.va_clean[ridge_nums + ridge_cats])
    pr_clean = ridge_clean.predict(split.va_clean[ridge_nums + ridge_cats])

    assert split.honest_metrics is not None
    return {
        "ridge_target_mentah": {
            "rmse": round(float(root_mean_squared_error(split.y_va_clean, pr_raw)), 4),
            "mae": round(float(mean_absolute_error(split.y_va_clean, pr_raw)), 4),
            "mean_bias": round(float(np.mean(pr_raw - split.y_va_clean)), 4),
            "pinball_p50": round(pinball_loss(split.y_va_clean, pr_raw, 0.5), 4),
        },
        "ridge_target_koreksi_60": {
            "rmse": round(float(root_mean_squared_error(split.y_va_clean, pr_clean)), 4),
            "mae": round(float(mean_absolute_error(split.y_va_clean, pr_clean)), 4),
            "mean_bias": round(float(np.mean(pr_clean - split.y_va_clean)), 4),
            "pinball_p50": round(pinball_loss(split.y_va_clean, pr_clean, 0.5), 4),
        },
        "lgbm_target_mentah": _hitung_metrik_kuantil(split.y_va_clean, p50_raw, p90_raw),
        "lgbm_target_koreksi_60": split.honest_metrics,
    }


def ukur_dampak_pemulihan_jarak(split: SplitDampak) -> dict[str, Any]:
    """Ukur dampak pemulihan sentinel ``jarak_km`` (0 & 9999) pada subset sentinel & global.

    Parameters
    ----------
    split : SplitDampak
        Objek split kronologis dari :func:`siapkan_split_dampak`.

    Returns
    -------
    dict[str, Any]
        Perbandingan metrik sebelum vs sesudah pemulihan jarak pada baris sentinel
        validasi dan seluruh baris validasi.
    """
    base_cols = list(CATEGORICAL_COLUMNS) + list(NAMED_NUMERIC_COLUMNS) + list(ANONYMOUS_F_COLUMNS)
    tr_raw_j = split.berat_imputer.transform(split.tr_raw, add_indicator=False)
    va_raw_j = split.berat_imputer.transform(split.va_raw, add_indicator=False)
    X_tr, [X_va] = align_categorical_dtypes(tr_raw_j[base_cols], [va_raw_j[base_cols]])
    p50_raw_j, p90_raw_j = _fit_predict_lgbm_quantile(X_tr, split.y_tr_clean, X_va)

    sent_mask = identify_corrupt_jarak_mask(split.va_raw["jarak_km"])
    assert split.honest_metrics is not None
    assert split.honest_pred_p50 is not None
    assert split.honest_pred_p90 is not None

    return {
        "n_sentinel_validasi": int(sent_mask.sum()),
        "subset_sentinel_mentah": _hitung_metrik_kuantil(
            split.y_va_clean[sent_mask], p50_raw_j[sent_mask], p90_raw_j[sent_mask]
        ),
        "subset_sentinel_dipulihkan": _hitung_metrik_kuantil(
            split.y_va_clean[sent_mask],
            split.honest_pred_p50[sent_mask],
            split.honest_pred_p90[sent_mask],
        ),
        "global_mentah": _hitung_metrik_kuantil(split.y_va_clean, p50_raw_j, p90_raw_j),
        "global_dipulihkan": split.honest_metrics,
    }


def ukur_dampak_imputasi_berat(split: SplitDampak) -> dict[str, float]:
    """Bandingkan akurasi imputasi ``berat_kg``: median global vs ``volume * densitas(layanan)``.

    Evaluasi dilakukan pada baris validasi yang ``berat_kg`` aslinya diketahui,
    baik secara global maupun khusus pada layanan ``kargo`` (missing rate 32,9%).

    Parameters
    ----------
    split : SplitDampak
        Objek split kronologis dari :func:`siapkan_split_dampak`.

    Returns
    -------
    dict[str, float]
        MAE (dalam kg) imputasi median global vs berbasis densitas layanan.
    """
    va_known = split.va_raw.loc[split.va_raw["berat_kg"].notna()].copy()
    y_berat = va_known["berat_kg"].to_numpy(dtype=np.float64)
    med_global = float(split.tr_raw["berat_kg"].median())

    pred_median = np.full(len(va_known), med_global, dtype=np.float64)
    dens = va_known["layanan"].map(split.berat_imputer.density_by_service).fillna(
        split.berat_imputer.global_density
    )
    pred_densitas = (va_known["volume_liter"] * dens).to_numpy(dtype=np.float64)
    is_kargo = (va_known["layanan"] == "kargo").to_numpy()

    return {
        "global_mae_median_kg": round(float(mean_absolute_error(y_berat, pred_median)), 3),
        "global_mae_densitas_kg": round(float(mean_absolute_error(y_berat, pred_densitas)), 3),
        "kargo_mae_median_kg": round(float(mean_absolute_error(y_berat[is_kargo], pred_median[is_kargo])), 3),
        "kargo_mae_densitas_kg": round(float(mean_absolute_error(y_berat[is_kargo], pred_densitas[is_kargo])), 3),
    }


def ukur_dampak_seleksi_f(split: SplitDampak) -> dict[str, dict[str, float]]:
    """Bandingkan model dengan semua 50 fitur ``f`` vs hanya 10 fitur sinyal ``f``.

    Menguji apakah membuang 40 fitur anonim noise (termasuk fitur drift noise
    ``f30``) memperbaiki generalisasi pada 2 bulan masa depan (Jul–Agu).

    Parameters
    ----------
    split : SplitDampak
        Objek split kronologis dari :func:`siapkan_split_dampak`.

    Returns
    -------
    dict[str, dict[str, float]]
        Metrik P50, P90, coverage, dan MAE untuk semua 50 fitur ``f`` vs 10 fitur sinyal.
    """
    pruned_cols = list(CATEGORICAL_COLUMNS) + list(NAMED_NUMERIC_COLUMNS) + list(SIGNAL_F_COLUMNS)
    X_tr, [X_va] = align_categorical_dtypes(split.tr_clean[pruned_cols], [split.va_clean[pruned_cols]])
    p50_pr, p90_pr = _fit_predict_lgbm_quantile(X_tr, split.y_tr_clean, X_va)

    assert split.honest_metrics is not None
    return {
        "semua_50_fitur_f": split.honest_metrics,
        "hanya_10_fitur_sinyal_f": _hitung_metrik_kuantil(split.y_va_clean, p50_pr, p90_pr),
    }


def plot_dampak_audit(hasil_dampak: dict[str, Any]) -> Path:
    """Buat grafik batang 2-panel dampak leakage ``jumlah_scan`` & pemulihan ``jarak_km``.

    Parameters
    ----------
    hasil_dampak : dict[str, Any]
        Keluaran dari :func:`jalankan_semua_dampak`.

    Returns
    -------
    Path
        Path file PNG yang tersimpan di ``kode/figures/task1_1a6_dampak_audit.png``.
    """
    ensure_dirs()
    configure_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    w = 0.36

    # Panel (a): Dampak leakage jumlah_scan
    e1 = hasil_dampak["exp1_leakage_jumlah_scan"]
    lbl1 = ["Dengan scan\n(CV naif)", "Dengan scan\n(simulasi test)", "Tanpa scan\n(jujur)"]
    p50_1 = [
        e1["dengan_scan_cv_naif"]["pinball_p50"],
        e1["dengan_scan_simulasi_test"]["pinball_p50"],
        e1["tanpa_scan_jujur"]["pinball_p50"],
    ]
    p90_1 = [
        e1["dengan_scan_cv_naif"]["pinball_p90"],
        e1["dengan_scan_simulasi_test"]["pinball_p90"],
        e1["tanpa_scan_jujur"]["pinball_p90"],
    ]
    x1 = np.arange(len(lbl1))
    b1 = axes[0].bar(x1 - w / 2, p50_1, width=w, color="#0284c7", label="Pinball P50")
    b2 = axes[0].bar(x1 + w / 2, p90_1, width=w, color="#f97316", label="Pinball P90")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(lbl1, fontsize=8.5)
    axes[0].set_ylabel("Pinball Loss (lebih kecil lebih baik)")
    axes[0].set_title("(a) Dampak Leakage jumlah_scan (Val Jul–Agu)")
    axes[0].set_ylim(0, max(p50_1 + p90_1) * 1.18)
    axes[0].legend(fontsize=8.5)

    # Panel (b): Dampak pemulihan sentinel jarak_km pada subset sentinel validasi
    e3 = hasil_dampak["exp3_pemulihan_jarak_km"]
    lbl2 = ["Pinball P50", "Pinball P90", "MAE P50"]
    raw_v = [
        e3["subset_sentinel_mentah"]["pinball_p50"],
        e3["subset_sentinel_mentah"]["pinball_p90"],
        e3["subset_sentinel_mentah"]["mae_p50"],
    ]
    rec_v = [
        e3["subset_sentinel_dipulihkan"]["pinball_p50"],
        e3["subset_sentinel_dipulihkan"]["pinball_p90"],
        e3["subset_sentinel_dipulihkan"]["mae_p50"],
    ]
    x2 = np.arange(len(lbl2))
    b3 = axes[1].bar(x2 - w / 2, raw_v, width=w, color="#dc2626", label="jarak_km mentah (0 / 9999)")
    b4 = axes[1].bar(x2 + w / 2, rec_v, width=w, color="#16a34a", label="jarak_km dipulihkan dari rute")
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(lbl2, fontsize=8.5)
    axes[1].set_ylabel("Error pada baris sentinel (jam)")
    axes[1].set_title(f"(b) Dampak Pemulihan jarak_km (n={e3['n_sentinel_validasi']} baris)")
    axes[1].set_ylim(0, max(raw_v + rec_v) * 1.2)
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1a6_dampak_audit.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def jalankan_semua_dampak(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Jalankan seluruh 5 eksperimen pengukuran dampak Langkah 1a.6b secara terurut.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, Any]
        Hasil seluruh 5 eksperimen dampak beserta informasi split dan path grafik.
    """
    split = siapkan_split_dampak(train)
    hasil = {
        "protokol_split": {
            "train": f"2025-01-01..2025-06-30 (n={len(split.tr_raw):,})",
            "validasi": f"2025-07-01..2025-08-31 (n={len(split.va_raw):,})",
        },
        "exp1_leakage_jumlah_scan": ukur_dampak_leakage_scan(split, test),
        "exp2_satuan_target": ukur_dampak_satuan_target(split),
        "exp3_pemulihan_jarak_km": ukur_dampak_pemulihan_jarak(split),
        "exp4_imputasi_berat_kg": ukur_dampak_imputasi_berat(split),
        "exp5_seleksi_fitur_f": ukur_dampak_seleksi_f(split),
    }
    fig_path = plot_dampak_audit(hasil)
    hasil["grafik_dampak"] = fig_path.name
    return hasil
