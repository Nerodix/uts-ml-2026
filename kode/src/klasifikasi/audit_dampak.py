"""Pengukuran dampak kuantitatif setiap temuan audit Tugas 2a (Klasifikasi Fraud).

Setiap temuan audit diukur dampaknya secara terkendali pada split kronologis
matang (Train Jan–Mei 2025 -> Validasi Jun–Jul 2025, horizon 2 bulan yang
meniru data uji Sep–Okt 2025) serta jendela Agustus untuk *label delay*:
1. ``ukur_dampak_leakage_review``     : Leakage ``flag_review_manual`` (CV bocor vs simulasi uji vs jujur).
2. ``ukur_dampak_label_delay``        : Dampak *right-censoring* Agustus pada evaluasi & probabilitas.
3. ``ukur_dampak_seleksi_v``          : Memangkas 42 fitur noise ``v`` (termasuk drift ``v15``).
4. ``ukur_dampak_indikator_kualitas`` : Penambahan indikator kualitas & pola nominal baris-tunggal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import FIGURES_DIR, FRAUD_REVIEW_COST_IDR, N_JOBS, SEED, siapkan_direktori
from src.klasifikasi.preprocess import (
    ANONYMOUS_V_COLUMNS,
    CATEGORICAL_COLUMNS,
    CLEAN_BASE_FEATURES,
    CLEAN_WITH_FLAGS_FEATURES,
    LEAKAGE_COLUMNS,
    NAMED_NUMERIC_COLUMNS,
    SIGNAL_V_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    samakan_tipe_kategori,
    tambah_flag_kualitas_data,
)
from src.utils import (
    LEBAR_GAMBAR,
    angka_id,
    atur_gaya_grafik,
    hitung_biaya_total_fraud,
    label_batang,
    miring,
    sumbu_koma,
)

AUDIT_TRAIN_END = pd.Timestamp("2025-05-31 23:59:59")
AUDIT_VAL_START = pd.Timestamp("2025-06-01 00:00:00")
AUDIT_VAL_END = pd.Timestamp("2025-07-31 23:59:59")


@dataclass
class SplitDampakKlasifikasi:
    """Wadah split kronologis matang Jan–Mei -> Jun–Jul beserta data Agustus."""

    tr_df: pd.DataFrame
    va_df: pd.DataFrame
    aug_df: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray
    nom_va: np.ndarray
    honest_metrics: dict[str, float] | None = None


def _probe_lgbm_params(seed: int = SEED) -> dict[str, Any]:
    """Parameter standar probe LightGBM biner agar seluruh eksperimen dampak setara."""
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


def _evaluasi_probabilitas(
    y_true: np.ndarray,
    prob: np.ndarray,
    nominal: np.ndarray,
) -> dict[str, float]:
    """Hitung PR-AUC, ROC-AUC, Brier score, dan total biaya aturan Bayes optimal."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y_true, dtype=np.int8)
    nom = np.asarray(nominal, dtype=np.float64)

    # Aturan keputusan Bayes untuk probabilitas: tandai jika p * nominal > 25.000
    flagged = (p * nom > FRAUD_REVIEW_COST_IDR).astype(np.int8)
    cost = hitung_biaya_total_fraud(y, flagged, nom, review_cost=FRAUD_REVIEW_COST_IDR)
    cost_none = hitung_biaya_total_fraud(y, np.zeros_like(y), nom, review_cost=FRAUD_REVIEW_COST_IDR)
    cost_all = hitung_biaya_total_fraud(y, np.ones_like(y), nom, review_cost=FRAUD_REVIEW_COST_IDR)
    n_flagged = int(flagged.sum())
    n_tp = int(((y == 1) & (flagged == 1)).sum())
    n_pos = int(y.sum())

    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_prob": float(np.mean(p)),
        "obs_rate": float(np.mean(y)),
        "ratio_pred_to_obs": float(np.mean(p) / max(float(np.mean(y)), 1e-12)),
        "total_cost_idr": float(cost),
        "cost_flag_none_idr": float(cost_none),
        "cost_flag_all_idr": float(cost_all),
        "cost_reduction_vs_none_pct": float((1.0 - cost / max(cost_none, 1.0)) * 100.0),
        "n_flagged": n_flagged,
        "flag_rate_pct": float(n_flagged / len(y) * 100.0),
        "recall_at_bayes": float(n_tp / max(n_pos, 1)),
    }


def _latih_dan_evaluasi_probe(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    fitur: Sequence[str],
    seed: int = SEED,
) -> tuple[lgb.LGBMClassifier, np.ndarray, dict[str, float]]:
    """Latih probe LightGBM pada daftar fitur tertentu dan kembalikan prediksi & metrik."""
    cat_in_fitur = [c for c in CATEGORICAL_COLUMNS if c in fitur]
    tr_al, [va_al] = samakan_tipe_kategori(tr_df, [va_df], cat_cols=cat_in_fitur)
    clf = lgb.LGBMClassifier(**_probe_lgbm_params(seed))
    clf.fit(tr_al[list(fitur)], tr_al[TARGET_COLUMN].to_numpy(dtype=np.int8))
    prob = clf.predict_proba(va_al[list(fitur)])[:, 1]
    metrik = _evaluasi_probabilitas(
        va_al[TARGET_COLUMN].to_numpy(dtype=np.int8),
        prob,
        va_al["nominal"].to_numpy(dtype=np.float64),
    )
    return clf, prob, metrik


def siapkan_split_dampak(train_raw: pd.DataFrame) -> SplitDampakKlasifikasi:
    """Bangun split kronologis matang Jan–Mei -> Jun–Jul dan subset Agustus.

    Parameters
    ----------
    train_raw : pd.DataFrame
        Data latih klasifikasi mentah (dengan atau tanpa kolom ``waktu_dt``).

    Returns
    -------
    SplitDampakKlasifikasi
        Wadah split latih, validasi matang 2 bulan, dan subset Agustus 2025.
    """
    df = tambah_flag_kualitas_data(train_raw)
    if "waktu_dt" not in df.columns:
        df["waktu_dt"] = pd.to_datetime(df[TIME_COLUMN])

    tr_df = df[df["waktu_dt"] <= AUDIT_TRAIN_END].copy()
    va_df = df[(df["waktu_dt"] >= AUDIT_VAL_START) & (df["waktu_dt"] <= AUDIT_VAL_END)].copy()
    aug_df = df[df["waktu_dt"] > AUDIT_VAL_END].copy()

    return SplitDampakKlasifikasi(
        tr_df=tr_df,
        va_df=va_df,
        aug_df=aug_df,
        y_tr=tr_df[TARGET_COLUMN].to_numpy(dtype=np.int8),
        y_va=va_df[TARGET_COLUMN].to_numpy(dtype=np.int8),
        nom_va=va_df["nominal"].to_numpy(dtype=np.float64),
    )


def ukur_dampak_leakage_review(
    split: SplitDampakKlasifikasi,
    seed: int = SEED,
) -> dict[str, Any]:
    """Ukur dampak leakage ``flag_review_manual`` (CV bocor vs simulasi uji vs jujur).

    Pada data uji Sep–Okt, ``flag_review_manual`` bernilai 1 pada ~1,23% transaksi
    secara acak (independen terhadap sinyal fraud, identik dengan laju 1,20% pada
    non-fraud data latih). Eksperimen ini mensimulasikan kondisi uji tersebut pada
    validasi Jun–Jul dengan mengganti ``flag_review_manual`` menjadi sampel acak
    Bernoulli dengan laju non-fraud data latih.

    Parameters
    ----------
    split : SplitDampakKlasifikasi
        Wadah split kronologis Jan–Mei -> Jun–Jul.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Metrik PR-AUC, ROC-AUC, dan total biaya pada skenario bocor naif,
        bocor di simulasi uji, serta model jujur tanpa ``flag_review_manual``.
    """
    fitur_jujur = list(CLEAN_BASE_FEATURES)
    fitur_bocor = fitur_jujur + list(LEAKAGE_COLUMNS)

    # 1) Model bocor dengan flag_review_manual
    clf_bocor, _, metrik_bocor_cv = _latih_dan_evaluasi_probe(
        split.tr_df, split.va_df, fitur_bocor, seed=seed
    )

    # 2) Evaluasi model bocor pada simulasi kondisi data uji (flag_review_manual acak ~1,20%)
    rng = np.random.default_rng(seed)
    p_bg_review = float(split.tr_df.loc[split.tr_df[TARGET_COLUMN] == 0, "flag_review_manual"].mean())
    va_sim = split.va_df.copy()
    va_sim["flag_review_manual"] = (rng.random(len(va_sim)) < p_bg_review).astype(np.int8)
    _, [va_sim_al] = samakan_tipe_kategori(split.tr_df, [va_sim], cat_cols=CATEGORICAL_COLUMNS)
    prob_sim = clf_bocor.predict_proba(va_sim_al[fitur_bocor])[:, 1]
    metrik_bocor_sim = _evaluasi_probabilitas(split.y_va, prob_sim, split.nom_va)

    # 3) Model jujur tanpa flag_review_manual
    _, _, metrik_jujur = _latih_dan_evaluasi_probe(
        split.tr_df, split.va_df, fitur_jujur, seed=seed
    )
    split.honest_metrics = metrik_jujur

    return {
        "laju_review_acak_non_fraud_train": p_bg_review,
        "dengan_review_cv_naif": metrik_bocor_cv,
        "dengan_review_simulasi_uji": metrik_bocor_sim,
        "jujur_tanpa_review": metrik_jujur,
        "penurunan_pr_auc_bocor_saat_uji": float(
            metrik_bocor_sim["pr_auc"] - metrik_bocor_cv["pr_auc"]
        ),
        "keunggulan_pr_auc_jujur_vs_bocor_uji": float(
            metrik_jujur["pr_auc"] - metrik_bocor_sim["pr_auc"]
        ),
        "penghematan_biaya_jujur_vs_bocor_uji_idr": float(
            metrik_bocor_sim["total_cost_idr"] - metrik_jujur["total_cost_idr"]
        ),
    }


def ukur_dampak_label_delay(
    train_raw: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Ukur distorsi *right-censoring* Agustus pada evaluasi validasi dan kalibrasi.

    Melatih satu model jujur pada Jan–Jun 2025 (100% matang), lalu membandingkan
    evaluasinya pada tiga jendela berurutan:
    - Jul 2025 (matang 100%, 31–61 hari sebelum snapshot).
    - Agt 1–31 2025 (campuran matang & tersensor).
    - Agt 15–31 2025 (tersensor berat, 0–16 hari sebelum snapshot).

    Selain itu mengukur depresi probabilitas rata-rata pada data uji jika model
    dilatih memakai seluruh Agustus tersensor dibandingkan hanya periode matang.

    Parameters
    ----------
    train_raw : pd.DataFrame
        Data latih klasifikasi penuh (Jan–Agt 2025).
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Perbandingan PR-AUC, rasio probabilitas prediksi terhadap laju teramati,
        dan biaya tampak antara jendela matang dan jendela tersensor.
    """
    df = train_raw.copy()
    if "waktu_dt" not in df.columns:
        df["waktu_dt"] = pd.to_datetime(df[TIME_COLUMN])

    fitur = list(CLEAN_BASE_FEATURES)
    tr_jj = df[df["waktu_dt"] <= pd.Timestamp("2025-06-30 23:59:59")].copy()
    val_jul = df[(df["waktu_dt"] >= pd.Timestamp("2025-07-01")) & (df["waktu_dt"] <= pd.Timestamp("2025-07-31 23:59:59"))].copy()
    val_aug_all = df[df["waktu_dt"] >= pd.Timestamp("2025-08-01")].copy()
    val_aug_late = df[df["waktu_dt"] >= pd.Timestamp("2025-08-15")].copy()
    val_aug_nonqris_late = val_aug_late[val_aug_late["kanal"] != "qris"].copy()

    tr_al, aligned_vals = samakan_tipe_kategori(
        tr_jj,
        [val_jul, val_aug_all, val_aug_late, val_aug_nonqris_late],
        cat_cols=CATEGORICAL_COLUMNS,
    )
    clf = lgb.LGBMClassifier(**_probe_lgbm_params(seed))
    clf.fit(tr_al[fitur], tr_al[TARGET_COLUMN].to_numpy(dtype=np.int8))

    out: dict[str, Any] = {}
    for nama, sub in zip(
        ["val_jul_matang", "val_agt_penuh_tersensor", "val_agt_akhir_15_31", "val_agt_akhir_non_qris"],
        aligned_vals,
    ):
        p = clf.predict_proba(sub[fitur])[:, 1]
        out[nama] = _evaluasi_probabilitas(
            sub[TARGET_COLUMN].to_numpy(dtype=np.int8),
            p,
            sub["nominal"].to_numpy(dtype=np.float64),
        )
        out[nama]["n_rows"] = int(len(sub))
        out[nama]["n_fraud"] = int(sub[TARGET_COLUMN].sum())

    # Eksperimen dampak depresi probabilitas jika negatif tersensor Agt 15–31 ikut dilatih
    tr_mature_jul = df[df["waktu_dt"] <= pd.Timestamp("2025-07-31 23:59:59")].copy()
    tr_with_censored_tail = pd.concat(
        [tr_mature_jul, df[df["waktu_dt"] >= pd.Timestamp("2025-08-15")]], ignore_index=True
    )
    eval_early_aug = df[
        (df["waktu_dt"] >= pd.Timestamp("2025-08-01"))
        & (df["waktu_dt"] <= pd.Timestamp("2025-08-07 23:59:59"))
        & (df["kanal"] != "qris")
    ].copy()
    _, _, met_mat_on_early = _latih_dan_evaluasi_probe(tr_mature_jul, eval_early_aug, fitur, seed=seed)
    _, _, met_cen_on_early = _latih_dan_evaluasi_probe(tr_with_censored_tail, eval_early_aug, fitur, seed=seed)
    out["latihan_matang_vs_ekor_tersensor_pada_agt_awal_non_qris"] = {
        "latih_matang_jan_jul": met_mat_on_early,
        "latih_campur_ekor_tersensor_15_31": met_cen_on_early,
        "depresi_mean_prob_pct": float(
            (met_cen_on_early["mean_prob"] / max(met_mat_on_early["mean_prob"], 1e-12) - 1.0) * 100.0
        ),
    }
    return out


def ukur_dampak_seleksi_v(
    split: SplitDampakKlasifikasi,
    seed: int = SEED,
) -> dict[str, Any]:
    """Ukur dampak memangkas 42 fitur anonim ``v`` noise (termasuk drift ``v15``).

    Membandingkan tiga himpunan fitur pada split matang Jan–Mei -> Jun–Jul:
    - Fitur bernama + seluruh 50 fitur ``v01–v50`` (mengandung ``v15`` drift & 41 noise).
    - Hanya fitur bernama (tanpa ``v`` sama sekali).
    - Fitur bernama + 8 fitur ``v`` bersinyal hasil screening audit.

    Parameters
    ----------
    split : SplitDampakKlasifikasi
        Wadah split kronologis matang Jan–Mei -> Jun–Jul.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Perbandingan metrik PR-AUC, ROC-AUC, dan total biaya ketiga konfigurasi.
    """
    named = list(CATEGORICAL_COLUMNS) + list(NAMED_NUMERIC_COLUMNS)
    all50 = named + list(ANONYMOUS_V_COLUMNS)
    sig8 = list(CLEAN_BASE_FEATURES)

    _, _, m_all50 = _latih_dan_evaluasi_probe(split.tr_df, split.va_df, all50, seed=seed)
    _, _, m_named = _latih_dan_evaluasi_probe(split.tr_df, split.va_df, named, seed=seed)
    if split.honest_metrics is not None:
        m_sig8 = split.honest_metrics
    else:
        _, _, m_sig8 = _latih_dan_evaluasi_probe(split.tr_df, split.va_df, sig8, seed=seed)

    return {
        "semua_50_v": m_all50,
        "hanya_bernama_tanpa_v": m_named,
        "bernama_plus_8_v_sinyal": m_sig8,
        "delta_pr_auc_8v_vs_50v": float(m_sig8["pr_auc"] - m_all50["pr_auc"]),
        "delta_biaya_8v_vs_50v_idr": float(m_sig8["total_cost_idr"] - m_all50["total_cost_idr"]),
        "signal_v_cols": list(SIGNAL_V_COLUMNS),
    }


def ukur_dampak_indikator_kualitas(
    split: SplitDampakKlasifikasi,
    seed: int = SEED,
) -> dict[str, Any]:
    """Ukur dampak penambahan 6 indikator kualitas data & pola nominal baris-tunggal.

    Dievaluasi pada validasi matang Jun–Jul (pola klasik) dan secara terpisah
    pada uji gelombang QRIS Agustus (latih Jan–Jul + Agt 1–10 -> evaluasi Agt 11–21)
    untuk mengukur dampak spesifik indikator ``is_nominal_bulat_100k``.

    Parameters
    ----------
    split : SplitDampakKlasifikasi
        Wadah split kronologis matang dan subset Agustus.
    seed : int, default=SEED
        Random seed deterministik.

    Returns
    -------
    dict[str, Any]
        Perbandingan metrik sebelum dan sesudah penambahan indikator kualitas.
    """
    m_base = (
        split.honest_metrics
        if split.honest_metrics is not None
        else _latih_dan_evaluasi_probe(split.tr_df, split.va_df, list(CLEAN_BASE_FEATURES), seed=seed)[2]
    )
    _, _, m_flags = _latih_dan_evaluasi_probe(
        split.tr_df, split.va_df, list(CLEAN_WITH_FLAGS_FEATURES), seed=seed
    )

    # Uji tambahan khusus gelombang QRIS baru (latih Jan–Jul + Agt 1–10 -> eval Agt 11–21 QRIS)
    tr_shift = pd.concat(
        [
            split.tr_df,
            split.va_df,
            split.aug_df[split.aug_df["waktu_dt"] <= pd.Timestamp("2025-08-10 23:59:59")],
        ],
        ignore_index=True,
    )
    va_shift = split.aug_df[
        (split.aug_df["waktu_dt"] > pd.Timestamp("2025-08-10 23:59:59"))
        & (split.aug_df["waktu_dt"] <= pd.Timestamp("2025-08-21 23:59:59"))
    ].copy()
    va_shift_qris = va_shift[va_shift["kanal"] == "qris"].copy()

    _, _, m_shift_base = _latih_dan_evaluasi_probe(
        tr_shift, va_shift_qris, list(CLEAN_BASE_FEATURES), seed=seed
    )
    _, _, m_shift_flags = _latih_dan_evaluasi_probe(
        tr_shift, va_shift_qris, list(CLEAN_WITH_FLAGS_FEATURES), seed=seed
    )

    return {
        "matang_jun_jul_dasar": m_base,
        "matang_jun_jul_dengan_flag": m_flags,
        "delta_pr_auc_matang": float(m_flags["pr_auc"] - m_base["pr_auc"]),
        "qris_shift_agt11_21_dasar": m_shift_base,
        "qris_shift_agt11_21_dengan_flag": m_shift_flags,
        "delta_pr_auc_qris_shift": float(m_shift_flags["pr_auc"] - m_shift_base["pr_auc"]),
    }


def simpan_plot_dampak_audit(
    hasil: dict[str, Any],
    output_path: Path = FIGURES_DIR / "task2_2a5_dampak_audit.png",
) -> Path:
    """Visualisasikan dampak kuantitatif temuan audit Tugas 2a dalam ukuran cetak laporan.

    Panel kiri menampilkan kehancuran PR-AUC dan pembengkakan total biaya bila fitur
    bocor ``flag_review_manual`` dipakai pada kondisi uji (review acak ~1,2%). Panel
    kanan menampilkan perbaikan PR-AUC akibat pemangkasan 42 fitur noise ``v`` dan
    penambahan indikator kualitas (terutama pada gelombang QRIS baru).

    Parameters
    ----------
    hasil : dict[str, Any]
        Keluaran dari ``jalankan_semua_dampak``.
    output_path : Path
        Lokasi penyimpanan berkas PNG beresolusi 300 DPI.

    Returns
    -------
    Path
        Path berkas gambar yang tersimpan.
    """
    atur_gaya_grafik()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.75))

    # Panel A: Dampak Leakage flag_review_manual (PR-AUC & Biaya dalam juta Rp)
    ax0 = axes[0]
    lk = hasil["leakage_review_manual"]
    label_a = ["Bocor\n(CV naif)", "Bocor\n(simulasi uji)", "Jujur\n(tanpa review)"]
    pr_vals = [
        lk["dengan_review_cv_naif"]["pr_auc"],
        lk["dengan_review_simulasi_uji"]["pr_auc"],
        lk["jujur_tanpa_review"]["pr_auc"],
    ]
    biaya_juta = [
        lk["dengan_review_cv_naif"]["total_cost_idr"] / 1e6,
        lk["dengan_review_simulasi_uji"]["total_cost_idr"] / 1e6,
        lk["jujur_tanpa_review"]["total_cost_idr"] / 1e6,
    ]
    x0 = np.arange(len(label_a))
    w = 0.36
    b1 = ax0.bar(x0 - w / 2, pr_vals, width=w, color=["#d62728", "#ff9896", "#2ca02c"], label="PR-AUC")
    ax0.set_xticks(x0)
    ax0.set_xticklabels(label_a)
    ax0.set_ylim(0, 1.18)
    ax0.set_ylabel("PR-AUC (lebih tinggi lebih baik)")
    sumbu_koma(ax0, "y")
    label_batang(ax0, b1, desimal=3)

    ax0_r = ax0.twinx()
    ax0_r.spines["top"].set_visible(False)
    ax0_r.grid(False)
    b2 = ax0_r.bar(
        x0 + w / 2,
        biaya_juta,
        width=w,
        color="#4c72b0",
        alpha=0.82,
        label="Biaya (juta Rp)",
    )
    ax0_r.set_ylim(0, max(biaya_juta) * 1.38)
    ax0_r.set_ylabel("Total biaya validasi (juta Rp)", color="#2f5597")
    ax0_r.tick_params(axis="y", colors="#2f5597")
    sumbu_koma(ax0_r, "y")
    for bar, val in zip(b2, biaya_juta):
        ax0_r.annotate(
            angka_id(val, 0),
            (bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 1.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="#2f5597",
        )
    ax0.set_title(f"(a) Dampak {miring('leakage')} flag_review_manual")
    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = ax0_r.get_legend_handles_labels()
    ax0.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=6.0)

    # Panel B: Dampak Seleksi 8 Fitur v & Indikator Kualitas pada PR-AUC
    ax1 = axes[1]
    sv = hasil["seleksi_fitur_v"]
    iq = hasil["indikator_kualitas"]
    label_b = [
        "Semua 50 v\n(Jun–Jul)",
        "Tanpa v\n(Jun–Jul)",
        "8 v sinyal\n(Jun–Jul)",
        "QRIS Agt\n(dasar)",
        "QRIS Agt\n(+flag bulat)",
    ]
    vals_b = [
        sv["semua_50_v"]["pr_auc"],
        sv["hanya_bernama_tanpa_v"]["pr_auc"],
        sv["bernama_plus_8_v_sinyal"]["pr_auc"],
        iq["qris_shift_agt11_21_dasar"]["pr_auc"],
        iq["qris_shift_agt11_21_dengan_flag"]["pr_auc"],
    ]
    colors_b = ["#8c8c8c", "#a6cee3", "#1f77b4", "#fdae6b", "#e6550d"]
    bars_b = ax1.bar(np.arange(len(label_b)), vals_b, color=colors_b, width=0.58)
    ax1.axvline(2.5, color="#666666", linestyle="--", linewidth=0.8)
    ax1.set_xticks(np.arange(len(label_b)))
    ax1.set_xticklabels(label_b, fontsize=6.4)
    ax1.set_ylim(0, max(vals_b) * 1.18)
    ax1.set_ylabel("PR-AUC")
    sumbu_koma(ax1, "y")
    label_batang(ax1, bars_b, desimal=3)
    ax1.set_title("(b) Seleksi fitur v & indikator pola nominal QRIS")

    fig.tight_layout(w_pad=1.6)
    siapkan_direktori()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def jalankan_semua_dampak(
    train_raw: pd.DataFrame,
    seed: int = SEED,
    simpan_plot: bool = True,
) -> dict[str, Any]:
    """Jalankan keempat eksperimen dampak audit Tugas 2a secara berurutan.

    Parameters
    ----------
    train_raw : pd.DataFrame
        Data latih klasifikasi mentah.
    seed : int, default=SEED
        Random seed deterministik.
    simpan_plot : bool, default=True
        Bila ``True``, menyimpan figur ``task2_2a5_dampak_audit.png``.

    Returns
    -------
    dict[str, Any]
        Ringkasan lengkap seluruh eksperimen dampak untuk ``task2_audit.json``.
    """
    split = siapkan_split_dampak(train_raw)
    hasil = {
        "split_info": {
            "train_period": "2025-01-01 s.d. 2025-05-31",
            "val_period": "2025-06-01 s.d. 2025-07-31",
            "n_train": int(len(split.tr_df)),
            "n_train_fraud": int(split.y_tr.sum()),
            "n_val": int(len(split.va_df)),
            "n_val_fraud": int(split.y_va.sum()),
        },
        "leakage_review_manual": ukur_dampak_leakage_review(split, seed=seed),
        "label_delay_censoring": ukur_dampak_label_delay(train_raw, seed=seed),
        "seleksi_fitur_v": ukur_dampak_seleksi_v(split, seed=seed),
        "indikator_kualitas": ukur_dampak_indikator_kualitas(split, seed=seed),
    }
    if simpan_plot:
        simpan_plot_dampak_audit(hasil)
    return hasil
