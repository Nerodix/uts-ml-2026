"""Analisis pergeseran pola fraud (gelombang QRIS Agustus) & adaptasi model (Tugas 2e).

Mengukur pergeseran distribusi pola fraud antar waktu (khususnya gelombang QRIS
bernominal bulat kelipatan Rp 100 rb yang muncul mulai 1 Agustus 2025) serta
menguji strategi adaptasi model secara kronologis pada jendela ``Agt 11–21``
(membuang ekor tersensor berat ``Agt 22–31``):

1. ``tanpa_adaptasi_jan_jul``      : Latih hanya pada Jan–Jul (buta terhadap gelombang QRIS).
2. ``tambah_agt_awal_seragam``     : Tambah Agt 1–10 berbobot 1×.
3. ``recency_weight_3x``           : Tambah Agt 1–10 berbobot 3×.
4. ``recency_weight_6x``           : Tambah Agt 1–10 berbobot 6×.
5. ``recency_weight_10x``          : Tambah Agt 1–10 berbobot 10×.
6. ``decay_dan_recency_6x``        : Peluruhan waktu ringan pada Jan–Jul + Agt 1–10 berbobot 6×.

Modul ini juga menyediakan ``buat_bobot_latih_produksi`` untuk pelatihan final
Sep–Okt dengan koreksi *right-censoring* berbasis kurva maturitas ``m(dt)``:
pada jendela parsial Agt 11–21, bobot ``y=1`` sebesar ``w_recent`` dan bobot
``y=0`` sebesar ``w_recent * m(dt)`` memulihkan rasio *odds* sebenarnya ``p/(1-p)``.

Jalankan langsung:
    python kode/src/klasifikasi/shift.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    FRAUD_REVIEW_COST_IDR,
    KLASIFIKASI_TEST_PATH,
    KLASIFIKASI_TRAIN_PATH,
    N_JOBS,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.klasifikasi.features import (  # noqa: E402
    SELECTED_TASK2_FEATURES,
    bangun_fitur_bebas_label,
)
from src.klasifikasi.preprocess import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    SNAPSHOT_TIMESTAMP,
    TARGET_COLUMN,
    align_categorical_dtypes,
    estimasi_kurva_maturitas_label,
    hitung_hari_sebelum_snapshot,
    muat_data_klasifikasi,
)
from src.klasifikasi.validation import buat_split_shift_agustus  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    configure_plotting,
    fraud_total_cost,
    label_batang,
    load_json,
    miring,
    save_json,
    set_global_seed,
    sumbu_koma,
    timer,
)

FEATURE_LIST = list(SELECTED_TASK2_FEATURES)


def _ambil_lgbm_params(seed: int = SEED) -> dict[str, Any]:
    """Muat konfigurasi terbaik LightGBM dari task2_models.json bila tersedia."""
    model_json = RESULTS_DIR / "task2_models.json"
    if model_json.exists():
        cfg = load_json(model_json).get("best_configs", {}).get("lgbm", {})
        if cfg:
            return {**cfg, "n_jobs": N_JOBS, "random_state": seed}
    return {
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }


def buat_bobot_latih_produksi(
    train_df: pd.DataFrame,
    w_recent: float = 6.0,
    decay_half_life_days: float | None = None,
    min_maturity_cut: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Bangun masker baris dan bobot latih produksi yang adaptif-shift & anti-censoring.

    Aturan pembobotan pada data latih Jan–Agt 2025:
    - **Jan 1 s.d. Jul 31 (matang 100%):** bobot dasar ``1,0`` (atau peluruhan waktu
      opsional bila ``decay_half_life_days`` diisi).
    - **Agt 1 s.d. Agt 10 (awal gelombang QRIS, matang ~90–100%):** bobot ``w_recent``
      untuk positif maupun negatif agar model menyerap pola QRIS baru.
    - **Agt 11 s.d. Agt 21 (tersensor parsial, ``m(dt) >= 0,25``):** positif terkonfirmasi
      berbobot ``w_recent``, sedangkan negatif berbobot ``w_recent * m(dt)`` sehingga rasio
      *odds* ``P(y=1)/P(y=0)`` tak bias terhadap *right-censoring*.
    - **Agt 22 s.d. Agt 31 (tersensor berat, ``m(dt) < 0,25``):** dikeluarkan (``keep=False``)
      karena 29–31 Agt bernilai 0 fraud terkonfirmasi.

    Parameters
    ----------
    train_df : pd.DataFrame
        Data latih klasifikasi lengkap dengan ``waktu_dt`` dan ``is_fraud``.
    w_recent : float, default=6.0
        Pengali bobot untuk periode gelombang QRIS Agustus.
    decay_half_life_days : float or None, default=None
        Waktu paruh peluruhan eksponensial (hari) pada Jan–Jul jika digunakan.
    min_maturity_cut : float, default=0.25
        Ambang minimum maturitas agar jendela parsial tetap disertakan.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(keep_mask, sample_weight)`` sepanjang ``train_df`` (baris dibuang bernilai 0).
    """
    w_dt = pd.to_datetime(train_df["waktu_dt"])
    y = train_df[TARGET_COLUMN].to_numpy(dtype=np.int8)
    dbs = hitung_hari_sebelum_snapshot(w_dt, SNAPSHOT_TIMESTAMP).to_numpy(dtype=np.float64)

    # Estimasi fungsi maturitas empiris m(dt) monoton dari kurva 2a
    kurva = estimasi_kurva_maturitas_label(train_df, SNAPSHOT_TIMESTAMP).sort_values("tengah_hari")
    xp = kurva["tengah_hari"].to_numpy(dtype=np.float64)
    fp = np.maximum.accumulate(kurva["rasio_maturitas"].to_numpy(dtype=np.float64))
    mat = np.interp(dbs, xp, fp, left=0.0, right=1.0)

    weights = np.ones(len(train_df), dtype=np.float64)
    if decay_half_life_days is not None and decay_half_life_days > 0:
        days_from_aug = np.maximum((pd.Timestamp("2025-08-01") - w_dt).dt.total_seconds().to_numpy() / 86_400.0, 0.0)
        weights = np.power(0.5, days_from_aug / float(decay_half_life_days))

    # Agt 1–10 (awal gelombang QRIS, dbs ~ 21–31h)
    aug_early = (w_dt >= pd.Timestamp("2025-08-01")) & (w_dt <= pd.Timestamp("2025-08-10 23:59:59"))
    weights[aug_early.to_numpy()] = float(w_recent)

    # Agt 11–21 (tersensor parsial, dbs ~ 10–21h): koreksi odds m(dt) pada negatif
    aug_mid = (w_dt > pd.Timestamp("2025-08-10 23:59:59")) & (w_dt <= pd.Timestamp("2025-08-21 23:59:59"))
    mid_mask = aug_mid.to_numpy()
    mat_mid = np.clip(mat[mid_mask], min_maturity_cut, 1.0)
    weights[mid_mask] = np.where(y[mid_mask] == 1, float(w_recent), float(w_recent) * mat_mid)

    # Agt 22–31 (tersensor berat, dbs < 10h): keluarkan dari pelatihan
    aug_tail = (w_dt > pd.Timestamp("2025-08-21 23:59:59")).to_numpy()
    weights[aug_tail] = 0.0
    keep_mask = ~aug_tail
    return keep_mask, weights


def _evaluasi_irisan_shift(
    y: np.ndarray,
    prob: np.ndarray,
    nom: np.ndarray,
    is_qris: np.ndarray,
) -> dict[str, float]:
    """Hitung PR-AUC & biaya pada keseluruhan, khusus QRIS, dan non-QRIS."""
    p = np.clip(prob, 1e-7, 1.0 - 1e-7)
    flg = (p * nom > FRAUD_REVIEW_COST_IDR).astype(np.int8)

    def _sub(mask: np.ndarray) -> tuple[float, float, float, float]:
        ym, pm, nm, fm = y[mask], p[mask], nom[mask], flg[mask]
        pr = float(average_precision_score(ym, pm)) if ym.sum() > 0 else 0.0
        roc = float(roc_auc_score(ym, pm)) if ym.sum() > 0 else 0.5
        cost = float(fraud_total_cost(ym, fm, nm, review_cost=FRAUD_REVIEW_COST_IDR))
        rec = float(((ym == 1) & (fm == 1)).sum() / max(int(ym.sum()), 1))
        return pr, roc, cost, rec

    pr_all, roc_all, cost_all, rec_all = _sub(np.ones(len(y), dtype=bool))
    pr_q, roc_q, cost_q, rec_q = _sub(is_qris)
    pr_nq, roc_nq, cost_nq, rec_nq = _sub(~is_qris)

    return {
        "overall_pr_auc": pr_all,
        "overall_roc_auc": roc_all,
        "overall_cost_idr": cost_all,
        "overall_recall": rec_all,
        "qris_pr_auc": pr_q,
        "qris_roc_auc": roc_q,
        "qris_cost_idr": cost_q,
        "qris_recall": rec_q,
        "non_qris_pr_auc": pr_nq,
        "non_qris_cost_idr": cost_nq,
        "non_qris_recall": rec_nq,
        "flag_rate_pct": float(flg.mean() * 100.0),
    }


def jalankan_eksperimen_shift(save_json_flag: bool = True) -> dict[str, Any]:
    """Jalankan eksperimen adaptasi *pattern shift* pada jendela kronologis Agt 11–21."""
    set_global_seed(SEED)
    ensure_dirs()

    with timer("Muat data & fitur kausal 2c untuk eksperimen shift 2e"):
        train_raw, _ = muat_data_klasifikasi(KLASIFIKASI_TRAIN_PATH, KLASIFIKASI_TEST_PATH)
        train_feat, _ = bangun_fitur_bebas_label(train_raw, None)
        splits = buat_split_shift_agustus(train_feat)

    idx_jj = splits["jan_jul_idx"]
    idx_ad = splits["aug_adapt_idx"]
    idx_va = splits["aug_val_idx"]

    df_jj = train_feat.iloc[idx_jj]
    df_jj_ad = train_feat.iloc[np.concatenate([idx_jj, idx_ad])]
    df_va = train_feat.iloc[idx_va]

    tr_jj, [va_for_jj] = align_categorical_dtypes(df_jj, [df_va], cat_cols=CATEGORICAL_COLUMNS)
    tr_ad, [va_for_ad] = align_categorical_dtypes(df_jj_ad, [df_va], cat_cols=CATEGORICAL_COLUMNS)

    y_va = va_for_ad[TARGET_COLUMN].to_numpy(dtype=np.int8)
    nom_va = va_for_ad["nominal"].to_numpy(dtype=np.float64)
    is_qris_va = (va_for_ad["kanal"].astype(str) == "qris").to_numpy()
    lgb_params = _ambil_lgbm_params(SEED)

    # Siapkan vektor bobot tiap strategi adaptasi pada tr_ad
    is_aug_ad = (pd.to_datetime(tr_ad["waktu_dt"]) >= pd.Timestamp("2025-08-01")).to_numpy()
    days_before_aug = np.maximum(
        (pd.Timestamp("2025-08-01") - pd.to_datetime(tr_ad["waktu_dt"])).dt.total_seconds().to_numpy() / 86_400.0,
        0.0,
    )
    decay_base = np.power(0.5, days_before_aug / 180.0)

    strategi_list: list[tuple[str, str, pd.DataFrame, pd.DataFrame, np.ndarray | None, float, float | None]] = [
        ("tanpa_adaptasi_jan_jul", "Tanpa adaptasi\n(Jan–Jul)", tr_jj, va_for_jj, None, 0.0, None),
        ("tambah_agt_awal_1x", "+Agt 1–10\n(bobot 1×)", tr_ad, va_for_ad, np.ones(len(tr_ad)), 1.0, None),
        ("recency_3x", "+Agt 1–10\n(bobot 3×)", tr_ad, va_for_ad, np.where(is_aug_ad, 3.0, 1.0), 3.0, None),
        ("recency_6x", "+Agt 1–10\n(bobot 6×)", tr_ad, va_for_ad, np.where(is_aug_ad, 6.0, 1.0), 6.0, None),
        ("recency_10x", "+Agt 1–10\n(bobot 10×)", tr_ad, va_for_ad, np.where(is_aug_ad, 10.0, 1.0), 10.0, None),
        ("decay_dan_recency_6x", "Decay 180h\n+Agt 6×", tr_ad, va_for_ad, np.where(is_aug_ad, 6.0, decay_base), 6.0, 180.0),
    ]

    hasil_strategi: list[dict[str, Any]] = []
    for kode, lbl, tr_x, va_x, sw, w_rec, hl in strategi_list:
        with timer(f"2e Evaluasi strategi: {kode}"):
            clf = lgb.LGBMClassifier(**lgb_params)
            clf.fit(tr_x[FEATURE_LIST], tr_x[TARGET_COLUMN].to_numpy(dtype=np.int8), sample_weight=sw)
            prob = clf.predict_proba(va_x[FEATURE_LIST])[:, 1]
            met = _evaluasi_irisan_shift(y_va, prob, nom_va, is_qris_va)
            met["strategi"] = kode
            met["label_pendek"] = lbl
            met["w_recent"] = w_rec
            met["decay_half_life_days"] = hl
            hasil_strategi.append(met)
            print(
                f"  Overall PR-AUC={met['overall_pr_auc']:.4f} | QRIS PR-AUC={met['qris_pr_auc']:.4f} "
                f"(rec={met['qris_recall']*100:.1f}%) | Non-QRIS PR-AUC={met['non_qris_pr_auc']:.4f} | "
                f"Cost={met['overall_cost_idr']/1e6:.2f}M Rp"
            )

    # Pilih strategi terbaik di antara kandidat adaptasi (bobot > 0) berdasarkan skor gabungan PR-AUC & biaya
    kandidat_adapt = [r for r in hasil_strategi if r["w_recent"] > 0]
    terbaik = min(kandidat_adapt, key=lambda r: r["overall_cost_idr"])

    # Ringkas statistik bobot produksi untuk data penuh Jan–Agt
    keep_prod, w_prod = buat_bobot_latih_produksi(
        train_feat,
        w_recent=terbaik["w_recent"],
        decay_half_life_days=terbaik["decay_half_life_days"],
    )

    simpan_plot_adaptasi_shift(hasil_strategi)
    hasil = {
        "jendela_evaluasi": {
            "periode": "2025-08-11 s.d. 2025-08-21",
            "n_total": int(len(y_va)),
            "n_fraud_total": int(y_va.sum()),
            "n_qris": int(is_qris_va.sum()),
            "n_fraud_qris": int(y_va[is_qris_va].sum()),
            "n_fraud_non_qris": int(y_va[~is_qris_va].sum()),
        },
        "strategi": [{k: v for k, v in r.items() if k != "label_pendek"} for r in hasil_strategi],
        "strategi_terpilih": {
            "nama": terbaik["strategi"],
            "w_recent": terbaik["w_recent"],
            "decay_half_life_days": terbaik["decay_half_life_days"],
            "overall_pr_auc": terbaik["overall_pr_auc"],
            "qris_pr_auc": terbaik["qris_pr_auc"],
            "qris_recall": terbaik["qris_recall"],
            "overall_cost_idr": terbaik["overall_cost_idr"],
            "penghematan_vs_tanpa_adaptasi_idr": float(
                hasil_strategi[0]["overall_cost_idr"] - terbaik["overall_cost_idr"]
            ),
        },
        "produksi_censoring_aware": {
            "n_baris_dipakai": int(keep_prod.sum()),
            "n_baris_ekor_dibuang": int((~keep_prod).sum()),
            "total_bobot": float(w_prod[keep_prod].sum()),
        },
    }
    if save_json_flag:
        save_json(hasil, RESULTS_DIR / "task2_shift.json")
        print("[SAVED] task2_shift.json")
    return hasil


def simpan_plot_adaptasi_shift(
    strategi: list[dict[str, Any]],
    output_path: Path = FIGURES_DIR / "task2_2e_pattern_shift_adaptation.png",
) -> Path:
    """Gambar figur 2 panel dampak adaptasi gelombang QRIS Agustus (PR-AUC & Total Biaya)."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.75))

    lbls = [r["label_pendek"] for r in strategi]
    x = np.arange(len(lbls))

    # Panel A: PR-AUC Khusus QRIS vs Overall pada Agt 11–21
    ax0 = axes[0]
    w = 0.36
    pr_q = [r["qris_pr_auc"] for r in strategi]
    pr_all = [r["overall_pr_auc"] for r in strategi]
    b1 = ax0.bar(x - w / 2, pr_q, width=w, color="#d62728", label="Khusus QRIS")
    b2 = ax0.bar(x + w / 2, pr_all, width=w, color="#1f77b4", label="Semua kanal")
    ax0.set_xticks(x)
    ax0.set_xticklabels(lbls, fontsize=5.6)
    ax0.set_ylim(0, max(max(pr_q), max(pr_all)) * 1.26)
    ax0.set_ylabel("PR-AUC (Agt 11–21)")
    ax0.set_title(f"(a) PR-AUC saat {miring('pattern shift')} QRIS")
    sumbu_koma(ax0, "y")
    label_batang(ax0, b1, desimal=3, ukuran=5.1, putar=True)
    label_batang(ax0, b2, desimal=3, ukuran=5.1, putar=True)
    ax0.legend(loc="lower right", fontsize=5.8)

    # Panel B: Total Biaya pada Agt 11–21 (juta Rp) — memisahkan biaya QRIS & non-QRIS
    ax1 = axes[1]
    cost_q = [r["qris_cost_idr"] / 1e6 for r in strategi]
    cost_nq = [r["non_qris_cost_idr"] / 1e6 for r in strategi]
    cost_tot = [r["overall_cost_idr"] / 1e6 for r in strategi]
    ax1.bar(x, cost_q, width=0.55, color="#e6550d", label="Kerugian & review QRIS")
    ax1.bar(x, cost_nq, width=0.55, bottom=cost_q, color="#6baed6", label="Kerugian & review non-QRIS")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, fontsize=5.6)
    ax1.set_ylim(0, max(cost_tot) * 1.24)
    ax1.set_ylabel("Total biaya Agt 11–21 (juta Rp)")
    ax1.set_title("(b) Penurunan total biaya lewat adaptasi")
    sumbu_koma(ax1, "y")
    for xi, tot in zip(x, cost_tot):
        ax1.annotate(
            f"{tot:.1f}".replace(".", ","),
            (xi, tot),
            xytext=(0, 2.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.6,
        )
    ax1.legend(loc="upper right", fontsize=5.8)

    fig.tight_layout(w_pad=1.5)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2e fraud pattern shift analysis and adaptation.")
    parser.add_argument("--no-save-json", action="store_true", help="Skip saving task2_shift.json.")
    args = parser.parse_args()
    jalankan_eksperimen_shift(save_json_flag=not args.no_save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
