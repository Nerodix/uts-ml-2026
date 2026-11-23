"""Pemecahan error, interpretasi SHAP, refit produksi adaptif, & prediksi_klasifikasi.csv (Tugas 2g).

Menjalankan empat langkah akhir Tugas 2:
1. Pemecahan error model terkalibrasi (Fold 3 Jul) per ``kanal``, ``kategori_merchant``,
   dan 5 arketipe pola fraud (serta evaluasi gelombang QRIS baru pada jendela shift Agt 11–21).
2. Penjelasan kontribusi fitur global dengan ``shap.TreeExplainer`` pada model LightGBM.
3. Pelatihan ulang *production ensemble* (LightGBM + CatBoost + XGBoost) pada aliran
   Jan–Agt 2025 dengan pembobotan adaptif-shift & anti-*right-censoring* (Tugas 2e)
   serta kalibrasi final (Tugas 2f).
4. Penulisan dan validasi otomatis berkas ``prediksi_klasifikasi.csv`` (106.674 baris).

Jalankan langsung:
    python kode/src/klasifikasi/analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    FRAUD_REVIEW_COST_IDR,
    KLASIFIKASI_TEST_PATH,
    KLASIFIKASI_TRAIN_PATH,
    N_JOBS,
    PRED_KLASIFIKASI_PATH,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.klasifikasi.audit import tetapkan_arketipe_fraud  # noqa: E402
from src.klasifikasi.calibration import (  # noqa: E402
    aturan_keputusan_bayes,
    buat_kalibrator,
)
from src.klasifikasi.features import (  # noqa: E402
    SELECTED_TASK2_FEATURES,
    bangun_fitur_bebas_label,
)
from src.klasifikasi.models import (  # noqa: E402
    fit_predict_catboost,
    fit_predict_lgbm,
    fit_predict_xgboost,
    gabung_prediksi_logit,
)
from src.klasifikasi.preprocess import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    align_categorical_dtypes,
    muat_data_klasifikasi,
)
from src.klasifikasi.shift import buat_bobot_latih_produksi  # noqa: E402
from src.klasifikasi.validation import buat_split_shift_agustus  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    configure_plotting,
    fraud_total_cost,
    load_json,
    save_json,
    set_global_seed,
    sumbu_koma,
    timer,
)

FEATURE_LIST = list(SELECTED_TASK2_FEATURES)


def _ringkas_grup(
    df_sub: pd.DataFrame,
    group_col: str,
    y: np.ndarray,
    prob: np.ndarray,
    tandai: np.ndarray,
    nom: np.ndarray,
) -> list[dict[str, Any]]:
    """Hitung metrik error, kerugian lolos, dan biaya review per nilai ``group_col``."""
    work = pd.DataFrame(
        {
            "grp": df_sub[group_col].astype(str).to_numpy(),
            "y": y,
            "prob": prob,
            "tandai": tandai,
            "nom": nom,
        }
    )
    rows: list[dict[str, Any]] = []
    for nama, g in work.groupby("grp"):
        yg = g["y"].to_numpy(dtype=np.int8)
        pg = g["prob"].to_numpy(dtype=np.float64)
        fg = g["tandai"].to_numpy(dtype=np.int8)
        ng = g["nom"].to_numpy(dtype=np.float64)

        n_fraud = int(yg.sum())
        tp = int(((yg == 1) & (fg == 1)).sum())
        fp = int(((yg == 0) & (fg == 1)).sum())
        fn = int(((yg == 1) & (fg == 0)).sum())
        missed_loss = float(ng[(yg == 1) & (fg == 0)].sum())
        rev_cost = float(FRAUD_REVIEW_COST_IDR * (fg == 1).sum())
        tot_cost = fraud_total_cost(yg, fg, ng, review_cost=FRAUD_REVIEW_COST_IDR)
        pr_auc = float(average_precision_score(yg, pg)) if (n_fraud > 0 and n_fraud < len(yg)) else None

        rows.append(
            {
                "grup": str(nama),
                "n_trx": int(len(yg)),
                "n_fraud": n_fraud,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pr_auc": pr_auc,
                "recall": float(tp / max(n_fraud, 1)) if n_fraud > 0 else None,
                "precision": float(tp / max(tp + fp, 1)) if (tp + fp) > 0 else None,
                "missed_loss_idr": missed_loss,
                "review_cost_idr": rev_cost,
                "total_cost_idr": float(tot_cost),
                "flag_rate_pct": float(fg.mean() * 100.0),
            }
        )
    rows.sort(key=lambda r: r["total_cost_idr"], reverse=True)
    return rows


def simpan_plot_error_breakdown(
    by_kanal: list[dict[str, Any]],
    by_arketipe: list[dict[str, Any]],
    output_path: Path = FIGURES_DIR / "task2_2g_error_breakdown.png",
) -> Path:
    """Gambar figur 2 panel pemecahan error per kanal dan per arketipe pola fraud."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.75), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # Panel A: Kerugian Lolos vs Biaya Review per Kanal (Fold 3 Jul)
    ax0 = axes[0]
    lbl_k = [r["grup"].upper() if r["grup"] in ("qris", "va") else r["grup"].title() for r in by_kanal]
    missed_k = [r["missed_loss_idr"] / 1e6 for r in by_kanal]
    rev_k = [r["review_cost_idr"] / 1e6 for r in by_kanal]
    tot_k = [r["total_cost_idr"] / 1e6 for r in by_kanal]
    x0 = np.arange(len(lbl_k))

    ax0.bar(x0, missed_k, width=0.56, color="#d62728", label="Kerugian fraud lolos")
    ax0.bar(x0, rev_k, width=0.56, bottom=missed_k, color="#4c72b0", label="Biaya tinjauan")
    ax0.set_xticks(x0)
    ax0.set_xticklabels(lbl_k, fontsize=6.4)
    ax0.set_ylim(0, max(tot_k) * 1.24)
    ax0.set_ylabel("Total biaya Fold 3 (juta Rp)")
    ax0.set_title("(a) Komposisi biaya per kanal")
    sumbu_koma(ax0, "y")
    for xi, tot in zip(x0, tot_k):
        ax0.annotate(
            angka_id(tot, 1),
            (xi, tot),
            xytext=(0, 2.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
    ax0.legend(loc="upper right", fontsize=5.8)

    # Panel B: Recall & Kerugian Lolos per 5 Arketipe Pola Fraud
    ax1 = axes[1]
    pendek = {
        "Merchant Kolusif (Nominal Besar)": "Merchant Kolusif",
        "Pengambilalihan Akun (ATO)": "ATO (Akun)",
        "Card Testing & Velocity": "Card Testing",
        "Anomali Merchant & Lainnya": "Anomali Merchant",
        "Gelombang QRIS": "Gelombang QRIS",
    }
    ark_plot = list(reversed(by_arketipe))
    y1 = np.arange(len(ark_plot))
    rec_a = [(r["recall"] or 0.0) * 100.0 for r in ark_plot]
    miss_a = [r["missed_loss_idr"] / 1e6 for r in ark_plot]
    bars1 = ax1.barh(y1, rec_a, color="#2ca02c", height=0.58)
    ax1.set_yticks(y1)
    ax1.set_yticklabels([pendek.get(r["grup"], r["grup"]) for r in ark_plot], fontsize=6.2)
    ax1.set_xlim(0, 128)
    ax1.set_xlabel("Recall deteksi pada aturan Bayes (%)")
    ax1.set_title("(b) Recall & kerugian lolos per arketipe")
    sumbu_koma(ax1, "x")
    for bar, rec_v, m_v in zip(bars1, rec_a, miss_a):
        ax1.annotate(
            f"{angka_id(rec_v, 1)}% ({angka_id(m_v, 1)} jt)",
            (rec_v, bar.get_y() + bar.get_height() / 2),
            xytext=(2.5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.5,
        )

    fig.tight_layout(w_pad=1.5)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def simpan_plot_shap_summary(
    shap_rows: list[dict[str, Any]],
    output_path: Path = FIGURES_DIR / "task2_2g_shap_summary.png",
) -> Path:
    """Gambar peringkat 15 fitur terpenting berdasarkan rerata ``|SHAP|`` di ruang log-odds."""
    configure_plotting()
    top15 = list(reversed(shap_rows[:15]))
    fig, ax = plt.subplots(figsize=(LEBAR_GAMBAR, 3.05))

    y_pos = np.arange(len(top15))
    vals = [r["mean_abs_shap"] for r in top15]
    cols = [
        "#e6550d"
        if r["fitur"].startswith("merch_")
        else ("#2ca02c" if r["fitur"].startswith("user_") else "#1f77b4")
        for r in top15
    ]
    bars = ax.barh(y_pos, vals, color=cols, height=0.66)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["fitur"] for r in top15], fontsize=6.5)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Rerata |SHAP| pada log-odds fraud (semakin besar semakin berpengaruh)")
    ax.set_title("15 pendorong risiko fraud utama (SHAP TreeExplainer pada LightGBM)")
    sumbu_koma(ax, "x")
    for bar, v in zip(bars, vals):
        ax.annotate(
            angka_id(v, 3),
            (v, bar.get_y() + bar.get_height() / 2),
            xytext=(2.5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.8,
        )

    # Legenda warna kelompok fitur
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#e6550d", label="Riwayat kausal merchant"),
            Patch(facecolor="#2ca02c", label="Riwayat kausal pengguna"),
            Patch(facecolor="#1f77b4", label="Fitur transaksi & interaksi domain"),
        ],
        loc="lower right",
        fontsize=6.0,
    )

    fig.tight_layout()
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def jalankan_analisis_dan_prediksi_final(save_json_flag: bool = True) -> dict[str, Any]:
    """Jalankan Tugas 2g: error breakdown, SHAP, refit produksi, dan ``prediksi_klasifikasi.csv``."""
    set_global_seed(SEED)
    ensure_dirs()

    models_meta = load_json(RESULTS_DIR / "task2_models.json")
    shift_meta = load_json(RESULTS_DIR / "task2_shift.json")
    cal_meta = load_json(RESULTS_DIR / "task2_calibration.json")
    f3_cache = np.load(CACHE_DIR / "task2_fold3_calibrated.npz")
    oof_cache = np.load(CACHE_DIR / "task2_oof.npz")

    with timer("Muat data & bangun fitur kausal bebas-label (Train + Test)"):
        train_raw, test_raw = muat_data_klasifikasi(KLASIFIKASI_TRAIN_PATH, KLASIFIKASI_TEST_PATH)
        train_feat, test_feat = bangun_fitur_bebas_label(train_raw, test_raw)
        assert test_feat is not None

    # 1) Pemecahan Error Fold 3 (Jul 2025)
    with timer("2g.1 Pemecahan error per kanal, kategori merchant, & 5 arketipe pola fraud"):
        idx_f3 = f3_cache["idx_f3"]
        y_f3 = f3_cache["y_f3"]
        nom_f3 = f3_cache["nom_f3"]
        p_f3_cal = f3_cache["prob_cal"]
        flg_f3 = f3_cache["tandai"]

        df_f3 = train_feat.iloc[idx_f3].copy()
        df_f3["arketipe"] = tetapkan_arketipe_fraud(df_f3)

        by_kanal = _ringkas_grup(df_f3, "kanal", y_f3, p_f3_cal, flg_f3, nom_f3)
        by_kat = _ringkas_grup(df_f3, "kategori_merchant", y_f3, p_f3_cal, flg_f3, nom_f3)
        by_ark = _ringkas_grup(df_f3, "arketipe", y_f3, p_f3_cal, flg_f3, nom_f3)

        # Evaluasi tambahan khusus gelombang QRIS baru pada jendela shift Agt 11–21 memakai strategi 2e
        sp = buat_split_shift_agustus(train_feat)
        tr_s = train_feat.iloc[np.concatenate([sp["jan_jul_idx"], sp["aug_adapt_idx"]])]
        va_s = train_feat.iloc[sp["aug_val_idx"]].copy()
        tr_sal, [va_sal] = align_categorical_dtypes(tr_s, [va_s], cat_cols=CATEGORICAL_COLUMNS)
        w_rec = float(shift_meta["strategi_terpilih"]["w_recent"])
        hl = shift_meta["strategi_terpilih"]["decay_half_life_days"]
        is_aug_s = (pd.to_datetime(tr_sal["waktu_dt"]) >= pd.Timestamp("2025-08-01")).to_numpy()
        sw_s = np.where(is_aug_s, w_rec, 1.0)
        if hl is not None:
            d_s = np.maximum(
                (pd.Timestamp("2025-08-01") - pd.to_datetime(tr_sal["waktu_dt"])).dt.total_seconds().to_numpy()
                / 86_400.0,
                0.0,
            )
            sw_s = np.where(is_aug_s, w_rec, np.power(0.5, d_s / float(hl)))

        lgb_cfg = {**models_meta["best_configs"]["lgbm"], "n_jobs": N_JOBS, "random_state": SEED}
        _, p_va_s = fit_predict_lgbm(tr_sal, va_sal, lgb_cfg, sample_weight=sw_s)
        va_sal["arketipe"] = tetapkan_arketipe_fraud(va_sal)
        flg_va_s = aturan_keputusan_bayes(p_va_s, va_sal["nominal"].to_numpy(dtype=np.float64))
        by_ark_shift = _ringkas_grup(
            va_sal,
            "arketipe",
            va_sal[TARGET_COLUMN].to_numpy(dtype=np.int8),
            p_va_s,
            flg_va_s,
            va_sal["nominal"].to_numpy(dtype=np.float64),
        )
        qris_shift_row = next((r for r in by_ark_shift if r["grup"] == "Gelombang QRIS"), None)

        simpan_plot_error_breakdown(by_kanal, by_ark)
        print("  Error Fold 3 per kanal:")
        for r in by_kanal:
            print(
                f"    {r['grup']:10s}: fraud={r['n_fraud']:3d}, rec={(r['recall'] or 0)*100:5.1f}%, "
                f"cost={r['total_cost_idr']/1e6:5.2f}M (missed={r['missed_loss_idr']/1e6:5.2f}M)"
            )
        if qris_shift_row is not None:
            print(
                f"  Gelombang QRIS (Agt 11–21 adaptif): fraud={qris_shift_row['n_fraud']}, "
                f"rec={(qris_shift_row['recall'] or 0)*100:.1f}%, PR-AUC={qris_shift_row['pr_auc']:.4f}"
            )

    # 2) Pelatihan Final Production Ensemble (Jan–Agt Censoring-Aware + Shift-Adaptive) -> Sep–Okt
    with timer("2g.2 Refit production ensemble (LGBM + CatBoost + XGBoost) adaptif shift & sensor"):
        keep_mask, w_prod = buat_bobot_latih_produksi(
            train_feat,
            w_recent=w_rec,
            decay_half_life_days=hl,
        )
        tr_prod = train_feat.loc[keep_mask].copy()
        sw_prod = w_prod[keep_mask]
        tr_prod_al, [te_prod_al] = align_categorical_dtypes(tr_prod, [test_feat], cat_cols=CATEGORICAL_COLUMNS)

        cb_cfg = {**models_meta["best_configs"]["catboost"], "thread_count": N_JOBS, "random_seed": SEED}
        xgb_cfg = {**models_meta["best_configs"]["xgboost"], "n_jobs": N_JOBS, "random_state": SEED}

        clf_lgb_prod, p_te_lgb = fit_predict_lgbm(tr_prod_al, te_prod_al, lgb_cfg, sample_weight=sw_prod)
        _, p_te_cb = fit_predict_catboost(tr_prod_al, te_prod_al, cb_cfg, sample_weight=sw_prod)
        _, p_te_xgb = fit_predict_xgboost(tr_prod_al, te_prod_al, xgb_cfg, sample_weight=sw_prod)

        w_map = models_meta["best_configs"]["ensemble"]["weights_final_all3"]
        w_vec = np.array([w_map["lgbm"], w_map["catboost"], w_map["xgboost"]], dtype=np.float64)
        p_te_raw = gabung_prediksi_logit([p_te_lgb, p_te_cb, p_te_xgb], w_vec)

        # Terapkan kalibrator final yang dilatih pada seluruh 3 fold OOF matang (Mei–Jul)
        cal_name = cal_meta["kalibrator_terpilih"]
        cal_final = buat_kalibrator(cal_name)
        if cal_final is not None:
            p_all3 = np.concatenate([oof_cache["ensemble_f1"], oof_cache["ensemble_f2"], oof_cache["ensemble_f3"]])
            y_all3 = np.concatenate([oof_cache["y_f1"], oof_cache["y_f2"], oof_cache["y_f3"]])
            cal_final.fit(p_all3, y_all3)
            p_te_final = cal_final.predict(p_te_raw)
        else:
            p_te_final = np.clip(p_te_raw, 1e-6, 1.0 - 1e-6)

        nom_te = test_raw["nominal"].to_numpy(dtype=np.float64)
        tandai_te = aturan_keputusan_bayes(p_te_final, nom_te, review_cost=FRAUD_REVIEW_COST_IDR)

    # 3) Interpretasi SHAP pada model produksi LightGBM
    with timer("2g.3 Hitung SHAP TreeExplainer (sampel 4.000 baris uji + fraud Fold 3)"):
        rng = np.random.default_rng(SEED)
        idx_sample = rng.choice(len(te_prod_al), size=min(4_000, len(te_prod_al)), replace=False)
        X_shap = te_prod_al.iloc[idx_sample][FEATURE_LIST]
        explainer = shap.TreeExplainer(clf_lgb_prod)
        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            sv = sv[1]
        sv_arr = np.asarray(sv, dtype=np.float64)
        mean_abs = np.mean(np.abs(sv_arr), axis=0)
        shap_order = np.argsort(mean_abs)[::-1]
        shap_rows = [
            {"fitur": FEATURE_LIST[i], "mean_abs_shap": float(mean_abs[i])}
            for i in shap_order[:20]
        ]
        simpan_plot_shap_summary(shap_rows)
        print("  Top 5 SHAP:", ", ".join(f"{r['fitur']}={r['mean_abs_shap']:.3f}" for r in shap_rows[:5]))

    # 4) Simpan & validasi prediksi_klasifikasi.csv
    with timer("2g.4 Tulis & validasi prediksi_klasifikasi.csv"):
        pred_df = pd.DataFrame(
            {
                ID_COLUMN: test_raw[ID_COLUMN].astype(str),
                "prob": np.round(np.clip(p_te_final, 0.0, 1.0), 6),
                "tandai": tandai_te.astype(np.int64),
            }
        )
        pred_df.to_csv(PRED_KLASIFIKASI_PATH, index=False, float_format="%.6f")
        assert len(pred_df) == len(test_raw) and pred_df["prob"].between(0.0, 1.0).all()
        assert set(pred_df["tandai"].unique()).issubset({0, 1})

        # Sanity check menyeluruh pada prediksi uji Sep–Okt
        is_qris_te = (test_raw["kanal"] == "qris").to_numpy()
        is_bulat_te = (nom_te % 100_000 == 0)
        qris_bulat_te = is_qris_te & is_bulat_te
        qris_norm_te = is_qris_te & (~is_bulat_te)

        sanity = {
            "n_rows": int(len(pred_df)),
            "prob_min": float(p_te_final.min()),
            "prob_median": float(np.median(p_te_final)),
            "prob_mean_pct": float(p_te_final.mean() * 100.0),
            "prob_q99": float(np.quantile(p_te_final, 0.99)),
            "prob_max": float(p_te_final.max()),
            "n_flagged": int(tandai_te.sum()),
            "flag_rate_pct": float(tandai_te.mean() * 100.0),
            "median_nominal_flagged_idr": float(np.median(nom_te[tandai_te == 1])) if tandai_te.any() else 0.0,
            "median_nominal_unflagged_idr": float(np.median(nom_te[tandai_te == 0])),
            "non_qris_mean_prob_pct": float(p_te_final[~is_qris_te].mean() * 100.0),
            "qris_normal_mean_prob_pct": float(p_te_final[qris_norm_te].mean() * 100.0),
            "qris_bulat100k_n": int(qris_bulat_te.sum()),
            "qris_bulat100k_mean_prob_pct": float(p_te_final[qris_bulat_te].mean() * 100.0),
            "qris_bulat100k_flag_rate_pct": float(tandai_te[qris_bulat_te].mean() * 100.0),
            "per_kanal_flag_rate_pct": {
                str(k): float(tandai_te[(test_raw["kanal"] == k).to_numpy()].mean() * 100.0)
                for k in sorted(test_raw["kanal"].unique())
            },
        }
        print(
            f"  VALIDASI LULUS | n={sanity['n_rows']:,}, mean_prob={sanity['prob_mean_pct']:.3f}%, "
            f"flagged={sanity['n_flagged']:,} ({sanity['flag_rate_pct']:.2f}%)"
        )
        print(
            f"  Sanity QRIS bulat 100rb (n={sanity['qris_bulat100k_n']}): "
            f"mean_prob={sanity['qris_bulat100k_mean_prob_pct']:.1f}%, "
            f"flagged={sanity['qris_bulat100k_flag_rate_pct']:.1f}% | "
            f"QRIS normal mean_prob={sanity['qris_normal_mean_prob_pct']:.3f}%"
        )

    hasil = {
        "error_fold3_per_kanal": by_kanal,
        "error_fold3_top10_kategori": by_kat[:10],
        "error_fold3_per_arketipe": by_ark,
        "error_shift_agt11_21_qris": qris_shift_row,
        "shap_top20": shap_rows,
        "produksi_info": {
            "kalibrator": cal_name,
            "w_recent": w_rec,
            "decay_half_life_days": hl,
            "n_train_rows": int(len(tr_prod)),
            "ensemble_weights": w_map,
        },
        "test_sanity": sanity,
    }
    if save_json_flag:
        save_json(hasil, RESULTS_DIR / "task2_analysis.json")
        print("[SAVED] task2_analysis.json & prediksi_klasifikasi.csv")
    return hasil


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2g error analysis, SHAP, and final fraud predictions.")
    parser.add_argument("--no-save-json", action="store_true", help="Skip saving task2_analysis.json.")
    args = parser.parse_args()
    jalankan_analisis_dan_prediksi_final(save_json_flag=not args.no_save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
