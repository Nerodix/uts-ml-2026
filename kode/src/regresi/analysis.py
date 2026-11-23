"""Analisis Error, Interpretasi SHAP, & Generasi Prediksi Test Tugas 1f (Regresi).

Modul ini menyelesaikan langkah terakhir Tugas 1:
1. Breakdown error (Pinball P50, Pinball P90, P90 Coverage, MAE, bias median)
   menurut ``layanan``, ``pulau_tujuan``, dan ``mitra_kurir`` pada OOF Fold 3
   (Jul–Agu 2025) yang telah dikalibrasi Conformal Prediction.
2. Interpretasi model dengan ``shap.TreeExplainer`` pada LightGBM Quantile
   (membandingkan penggerak utama P50 vs P90 beserta arah efeknya).
3. Pelatihan final ketiga anggota aktif Ensemble (CatBoost MultiQuantile +
   LightGBM Quantile + Log-Target Ridge) pada 100% data latih (Jan–Agu 2025),
   penerapan bobot optimal Time CV + kalibrasi Group-Multiplicative CQR,
   penulisan ``prediksi_regresi.csv``, verifikasi kontrak otomatis, serta
   audit kewajaran prediksi pada ``K13`` (unseen) dan tanggal kembar ``9.9``/``10.10``.

Jalankan langsung:
    python kode/src/regresi/analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    PRED_REGRESI_PATH,
    REGRESI_TEST_PATH,
    REGRESI_TRAIN_PATH,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.regresi.features import Task1FeaturePipeline  # noqa: E402
from src.regresi.models import (  # noqa: E402
    LogTargetRidgeQuantile,
    fit_predict_catboost,
    fit_predict_lgbm,
)
from src.regresi.preprocess import DATE_COLUMN, ID_COLUMN, TARGET_COLUMN, fix_target_unit_errors  # noqa: E402
from src.regresi.validation import buat_fold_waktu  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    configure_plotting,
    label_batang,
    load_json,
    miring,
    p90_coverage,
    pinball_loss,
    save_json,
    set_global_seed,
    sumbu_koma,
    timer,
)


def _ringkas_grup(
    df: pd.DataFrame,
    group_col: str,
    y: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> list[dict[str, Any]]:
    """Hitung tabel metrik error lengkap per kategori pada ``group_col``."""
    rows: list[dict[str, Any]] = []
    for val, grp_idx in df.groupby(group_col).groups.items():
        idx = np.asarray(grp_idx)
        yt, q50, q90 = y[idx], p50[idx], p90[idx]
        rows.append(
            {
                group_col: str(val),
                "n": int(len(idx)),
                "median_aktual": round(float(np.median(yt)), 2),
                "median_p50": round(float(np.median(q50)), 2),
                "bias_median": round(float(np.median(q50 - yt)), 2),
                "mae_p50": round(float(mean_absolute_error(yt, q50)), 2),
                "pinball_p50": round(pinball_loss(yt, q50, 0.5), 4),
                "pinball_p90": round(pinball_loss(yt, q90, 0.9), 4),
                "p90_coverage": round(p90_coverage(yt, q90), 4),
            }
        )
    rows.sort(key=lambda r: r["pinball_p50"], reverse=True)
    return rows


def analisis_error_per_segmen(train: pd.DataFrame) -> dict[str, Any]:
    """Hitung breakdown error Fold 3 terkalibrasi menurut layanan, pulau, dan kurir.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.

    Returns
    -------
    dict[str, Any]
        Tabel error per ``layanan``, ``pulau_tujuan``, dan ``mitra_kurir`` beserta figur.
    """
    cal_cache = CACHE_DIR / "task1_fold3_calibrated.npz"
    if not cal_cache.exists():
        raise FileNotFoundError(
            f"{cal_cache} belum ada. Jalankan `models.py` lalu `calibration.py` terlebih dahulu."
        )
    d = np.load(cal_cache)
    df_f3 = train.iloc[d["idx_fold3"]].reset_index(drop=True)
    y_f3 = d["y_fold3"]
    p50 = d["p50_fold3"]
    p90 = d["p90_cal_fold3"]

    breakdown = {
        "overall_fold3_calibrated": {
            "n": int(len(y_f3)),
            "pinball_p50": round(pinball_loss(y_f3, p50, 0.5), 4),
            "pinball_p90": round(pinball_loss(y_f3, p90, 0.9), 4),
            "p90_coverage": round(p90_coverage(y_f3, p90), 4),
            "mae_p50": round(float(mean_absolute_error(y_f3, p50)), 4),
        },
        "by_layanan": _ringkas_grup(df_f3, "layanan", y_f3, p50, p90),
        "by_pulau_tujuan": _ringkas_grup(df_f3, "pulau_tujuan", y_f3, p50, p90),
        "by_mitra_kurir": _ringkas_grup(df_f3, "mitra_kurir", y_f3, p50, p90),
    }
    fig_path = plot_breakdown_error(breakdown)
    breakdown["figure"] = fig_path.name
    return breakdown


def plot_breakdown_error(breakdown: dict[str, Any]) -> Path:
    """Buat grafik 3-panel pinball loss menurut layanan, pulau tujuan, dan mitra kurir."""
    ensure_dirs()
    configure_plotting()

    # Tata letak "constrained" memberi ruang label per panel (nama pulau lebih panjang daripada kode kurir)
    fig, axes = plt.subplots(1, 3, figsize=(LEBAR_GAMBAR, 2.2), layout="constrained")
    w = 0.42

    specs = [
        (axes[0], breakdown["by_layanan"], "layanan", "(a) Menurut Layanan"),
        (axes[1], breakdown["by_pulau_tujuan"], "pulau_tujuan", "(b) Menurut Pulau Tujuan"),
        (axes[2], breakdown["by_mitra_kurir"], "mitra_kurir", "(c) Menurut Mitra Kurir"),
    ]
    for ax, rows, key, title in specs:
        labels = [str(r[key]) for r in rows]
        if key == "layanan":
            labels = [lbl.capitalize() for lbl in labels]
        v50 = [r["pinball_p50"] for r in rows]
        v90 = [r["pinball_p90"] for r in rows]
        y_pos = np.arange(len(labels))
        b50 = ax.barh(y_pos - w / 2, v50, height=w, color="#2563eb", label="P50")
        b90 = ax.barh(y_pos + w / 2, v90, height=w, color="#f97316", label="P90")
        # Panel kurir terlalu rapat untuk diberi angka; nilainya tercantum pada tabel lampiran laporan.
        if len(rows) <= 8:
            label_batang(ax, b50, mendatar=True, ukuran=5.8)
            label_batang(ax, b90, mendatar=True, ukuran=5.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel(f"{miring('Pinball loss')} (jam)")
        ax.set_title(title)
        ax.set_xlim(0, max(v50 + v90) * 1.25)
        ax.grid(axis="y", visible=False)
        sumbu_koma(ax, "x")

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=2, frameon=False)
    out_path = FIGURES_DIR / "task1_1f_error_breakdown.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def analisis_shap_model(train: pd.DataFrame, sample_size: int = 3_000) -> dict[str, Any]:
    """Hitung nilai SHAP pada LightGBM Quantile (P50 & P90) terbaik di Fold 3.

    Karena model dilatih pada skala ``log(durasi_jam)``, nilai SHAP langsung
    dapat diinterpretasikan secara intuitif sebagai kontribusi multiplikatif
    relatif (mis. ``SHAP = +0,20`` berarti menaikkan prediksi durasi sekitar
    ``exp(0,20) - 1 ≈ +22%``).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.
    sample_size : int, default=3000
        Jumlah sampel validasi Fold 3 untuk perhitungan ``TreeExplainer``.

    Returns
    -------
    dict[str, Any]
        Daftar 15 fitur paling berpengaruh untuk P50 dan P90 beserta nama grafik.
    """
    models_meta = load_json(RESULTS_DIR / "task1_models.json")
    lgb_cfg = models_meta["models"]["lgbm"]["best_config"]

    fold3 = buat_fold_waktu(train)[-1]
    tr_raw = train.iloc[fold3.train_idx].reset_index(drop=True)
    va_raw = train.iloc[fold3.val_idx].reset_index(drop=True)
    y_tr, _ = fix_target_unit_errors(tr_raw[TARGET_COLUMN])

    pipe = Task1FeaturePipeline(seed=SEED)
    X_tr = pipe.fit_transform(tr_raw, y_tr)
    X_va = pipe.transform(va_raw)

    _, _, m50, m90 = fit_predict_lgbm(
        X_tr,
        y_tr,
        X_va,
        params=lgb_cfg["params"],
        use_log_target=lgb_cfg["use_log_target"],
        drop_mitra_id=lgb_cfg["drop_mitra_id"],
        seed=SEED,
    )
    cols = [c for c in X_va.columns if not (lgb_cfg["drop_mitra_id"] and c == "mitra_kurir")]
    X_samp = X_va[cols].sample(min(sample_size, len(X_va)), random_state=SEED)

    exp50 = shap.TreeExplainer(m50)
    exp90 = shap.TreeExplainer(m90)
    sv50 = np.asarray(exp50.shap_values(X_samp), dtype=np.float64)
    sv90 = np.asarray(exp90.shap_values(X_samp), dtype=np.float64)

    mean_abs_50 = pd.Series(np.mean(np.abs(sv50), axis=0), index=cols).sort_values(ascending=False)
    mean_abs_90 = pd.Series(np.mean(np.abs(sv90), axis=0), index=cols).sort_values(ascending=False)

    top15 = list(mean_abs_50.head(15).index)

    # Grafik 2-panel: (a) rata-rata |SHAP| 15 fitur terpenting; (b) arah pengaruh 10 fitur numerik pada P50
    ensure_dirs()
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.3), gridspec_kw={"width_ratios": [1.0, 1.12]})

    y_pos = np.arange(len(top15))
    w = 0.4
    axes[0].barh(y_pos - w / 2, [mean_abs_50[f] for f in top15], height=w, color="#2563eb", label="Model P50")
    axes[0].barh(y_pos + w / 2, [mean_abs_90[f] for f in top15], height=w, color="#f97316", label="Model P90")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top15, fontsize=6.0)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Rata-rata nilai mutlak SHAP")
    axes[0].set_title("(a) Lima Belas Fitur Terpenting")
    axes[0].grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")

    # Panel (b): sebaran nilai SHAP per paket (warna = nilai fitur rendah sampai tinggi)
    num_top10 = [f for f in top15 if pd.api.types.is_numeric_dtype(X_samp[f])][:10]
    rng = np.random.default_rng(SEED)
    for rank, feat in enumerate(num_top10):
        f_idx = cols.index(feat)
        sv = sv50[:, f_idx]
        vals = X_samp[feat].to_numpy(dtype=np.float64)
        lo, hi = np.nanpercentile(vals, [2, 98])
        norm_v = np.clip((vals - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        jitter = rng.uniform(-0.22, 0.22, size=len(sv))
        sc = axes[1].scatter(
            sv,
            np.full(len(sv), rank) + jitter,
            c=norm_v,
            cmap="coolwarm",
            s=2.5,
            alpha=0.45,
            edgecolors="none",
            rasterized=True,
        )
    axes[1].axvline(0.0, color="#111827", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(num_top10)))
    axes[1].set_yticklabels(num_top10, fontsize=6.0)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Nilai SHAP (positif: durasi lebih lama)")
    axes[1].set_title("(b) Arah Pengaruh pada Model P50")
    axes[1].grid(axis="y", visible=False)
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.04, pad=0.02)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Rendah", "Tinggi"])
    cbar.set_label("Nilai fitur", labelpad=-8)
    cbar.outline.set_linewidth(0.4)
    for ax in axes:
        sumbu_koma(ax, "x")

    fig.tight_layout()
    fig_path = FIGURES_DIR / "task1_1f_shap_summary.png"
    fig.savefig(fig_path)
    plt.close(fig)

    return {
        "n_samples": int(len(X_samp)),
        "top15_p50_mean_abs_shap": {k: round(float(v), 5) for k, v in mean_abs_50.head(15).items()},
        "top15_p90_mean_abs_shap": {k: round(float(v), 5) for k, v in mean_abs_90.head(15).items()},
        "figure": fig_path.name,
    }


def latih_final_dan_prediksi_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Latih Ensemble final pada 100% train (Jan–Agu), kalibrasi CQR, & tulis ``prediksi_regresi.csv``.

    Menggunakan konfigurasi terbaik dan bobot konveks hasil Time CV dari
    ``task1_models.json``, serta faktor kalibrasi ``GroupMultiplicativeCQR``
    hasil ``task1_calibration.json`` (dipelajari pada OOF Fold 3 Jul–Agu).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih penuh ``regresi_train.csv.gz`` (250.000 baris).
    test : pd.DataFrame
        Data uji ``regresi_test.csv.gz`` (60.000 baris).

    Returns
    -------
    dict[str, Any]
        Ringkasan statistik prediksi test, uji kewajaran ``K13`` & tanggal kembar
        ``9.9``/``10.10``, serta status validator kontrak.
    """
    models_meta = load_json(RESULTS_DIR / "task1_models.json")
    cal_meta = load_json(RESULTS_DIR / "task1_calibration.json")

    ridge_cfg = models_meta["models"]["log_ridge"]["best_config"]
    lgb_cfg = models_meta["models"]["lgbm"]["best_config"]
    cb_cfg = models_meta["models"]["catboost"]["best_config"]
    w50 = models_meta["models"]["ensemble"]["weights_p50_final_all3"]
    w90 = models_meta["models"]["ensemble"]["weights_p90_final_all3"]
    cal_params = cal_meta["final_calibration_params"]

    y_tr_clean, _ = fix_target_unit_errors(train[TARGET_COLUMN])
    pipe = Task1FeaturePipeline(seed=SEED)
    X_tr = pipe.fit_transform(train, y_tr_clean)
    X_te = pipe.transform(test)

    # 1. Log-Target Ridge
    ridge_model = LogTargetRidgeQuantile(alpha=ridge_cfg["alpha"], seed=SEED).fit(X_tr, y_tr_clean)
    p50_r, p90_r = ridge_model.predict(X_te)

    # 2. LightGBM Quantile terbaik
    p50_l, p90_l, _, _ = fit_predict_lgbm(
        X_tr,
        y_tr_clean,
        X_te,
        params=lgb_cfg["params"],
        use_log_target=lgb_cfg["use_log_target"],
        drop_mitra_id=lgb_cfg["drop_mitra_id"],
        seed=SEED,
    )

    # 3. CatBoost MultiQuantile terbaik
    p50_c, p90_c, _ = fit_predict_catboost(
        X_tr,
        y_tr_clean,
        X_te,
        params=cb_cfg["params"],
        use_log_target=cb_cfg["use_log_target"],
        drop_mitra_id=cb_cfg["drop_mitra_id"],
        seed=SEED,
    )

    # Gabungkan sesuai bobot konveks optimal (normalisasi untuk keamanan numerik)
    s50 = w50["log_ridge"] + w50["lgbm"] + w50["catboost"]
    s90 = w90["log_ridge"] + w90["lgbm"] + w90["catboost"]
    p50_ens = (w50["log_ridge"] * p50_r + w50["lgbm"] * p50_l + w50["catboost"] * p50_c) / s50
    p90_raw = (w90["log_ridge"] * p90_r + w90["lgbm"] * p90_l + w90["catboost"] * p90_c) / s90

    # Terapkan kalibrasi CQR terpilih
    method = cal_params["selected_method"]
    if method == "group_multiplicative_cqr":
        factors_map = cal_params["group_multiplicative_factors_jul_agu"]
        g_factor = cal_params["multiplicative_factor_jul_agu"]
        fac = test["layanan"].astype(str).map(factors_map).fillna(g_factor).to_numpy(dtype=np.float64)
        p90_cal = np.maximum(p50_ens, p90_raw * fac)
    elif method == "multiplicative_cqr":
        p90_cal = np.maximum(p50_ens, p90_raw * cal_params["multiplicative_factor_jul_agu"])
    else:
        p90_cal = np.maximum(p50_ens, p90_raw + cal_params["additive_delta_jul_agu"])

    sub = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN],
            "p50": np.round(p50_ens, 4),
            "p90": np.round(np.maximum(p50_ens, p90_cal), 4),
        }
    )
    sub.to_csv(PRED_REGRESI_PATH, index=False)

    assert len(sub) == len(test) and (sub["p90"] >= sub["p50"]).all() and sub[["p50", "p90"]].notna().all().all()

    # Diagnostik kewajaran prediksi pada data uji Sep–Okt 2025
    dt_te = pd.to_datetime(test[DATE_COLUMN])
    is_k13 = test["mitra_kurir"] == "K13"
    is_k07 = test["mitra_kurir"] == "K07"
    is_k08 = test["mitra_kurir"] == "K08"
    is_twin_te = dt_te.dt.month == dt_te.dt.day

    return {
        "output_file": PRED_REGRESI_PATH.name,
        "n_rows": int(len(sub)),
        "validation_passed": True,
        "calibration_applied": method,
        "p50_quantiles": {str(q): round(float(v), 2) for q, v in sub["p50"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()},
        "p90_quantiles": {str(q): round(float(v), 2) for q, v in sub["p90"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()},
        "median_interval_width": round(float(np.median(sub["p90"] - sub["p50"])), 2),
        "sanity_checks_test": {
            "K08_cepat_median_p50": round(float(sub.loc[is_k08, "p50"].median()), 2),
            "other_median_p50": round(float(sub.loc[~(is_k13 | is_k07 | is_k08), "p50"].median()), 2),
            "K13_unseen_median_p50": round(float(sub.loc[is_k13, "p50"].median()), 2),
            "K07_shift_median_p50": round(float(sub.loc[is_k07, "p50"].median()), 2),
            "twin_99_1010_median_p50": round(float(sub.loc[is_twin_te, "p50"].median()), 2),
            "normal_days_median_p50": round(float(sub.loc[~is_twin_te, "p50"].median()), 2),
        },
    }


def main() -> None:
    """Jalankan seluruh Tugas 1f: breakdown error, SHAP, dan generasi ``prediksi_regresi.csv``."""
    set_global_seed(SEED)
    ensure_dirs()

    train = pd.read_csv(REGRESI_TRAIN_PATH)
    test = pd.read_csv(REGRESI_TEST_PATH)

    with timer("Analisis breakdown error (layanan, pulau_tujuan, mitra_kurir)"):
        breakdown = analisis_error_per_segmen(train)

    with timer("Analisis SHAP pada LightGBM Quantile P50 & P90"):
        shap_info = analisis_shap_model(train)

    with timer("Pelatihan Ensemble final (100% train) & generasi prediksi_regresi.csv"):
        sub_info = latih_final_dan_prediksi_test(train, test)

    report = {
        "task": "Tugas 1f — Analisis Error, Interpretasi SHAP, & Prediksi Final Regresi",
        "seed": SEED,
        "error_breakdown": breakdown,
        "shap_analysis": shap_info,
        "submission_summary": sub_info,
    }
    save_json(report, RESULTS_DIR / "task1_analysis.json")

    print("\n================ RINGKASAN TUGAS 1f ================")
    print("Breakdown Error Fold 3 Terkalibrasi (by layanan):")
    for r in breakdown["by_layanan"]:
        print(
            f"  {r['layanan']:10s} (n={r['n']:>6,}) -> P50={r['pinball_p50']:.4f} | "
            f"P90={r['pinball_p90']:.4f} | cov90={r['p90_coverage']*100:.1f}% | bias={r['bias_median']:+.2f}"
        )
    print("\nBreakdown Error Fold 3 Terkalibrasi (by pulau_tujuan):")
    for r in breakdown["by_pulau_tujuan"]:
        print(
            f"  {r['pulau_tujuan']:20s} (n={r['n']:>6,}) -> P50={r['pinball_p50']:.4f} | "
            f"P90={r['pinball_p90']:.4f} | cov90={r['p90_coverage']*100:.1f}%"
        )
    print("\nTop-8 fitur SHAP P50:", list(shap_info["top15_p50_mean_abs_shap"].items())[:8])
    print("\nUji kewajaran prediksi_regresi.csv (Sep–Okt 2025):")
    for k, v in sub_info["sanity_checks_test"].items():
        print(f"  {k:28s}: {v} jam")
    print(f"\nVALIDASI {sub_info['output_file']}: LULUS ({sub_info['n_rows']:,} baris)")


if __name__ == "__main__":
    main()
