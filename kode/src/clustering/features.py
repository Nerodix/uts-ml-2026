"""Seleksi fitur berbasis bukti kuantitatif & pembobotan blok Tugas 3b (Segmentasi Pelanggan).

Modul ini:
1. Menyaring seluruh 59 fitur kandidat pada pelanggan non-bot (118.200 baris) memakai
   tiga bukti independen: koherensi multivariat Spearman ``sum_r2``, prediktabilitas
   silang *out-of-fold* Ridge ``R^2``, dan koefisien bimodalitas Sarle (``BC``).
2. Mengungkap 21 fitur *noise* murni:
   - 15 kolom anonim ``n01–n15`` (pola 7 keluarga distribusi sintetis berulang, termasuk
     jebakan bimodal berbentuk-U ``n06`` & ``n13`` dengan ``BC ≈ 0,666`` namun ``R^2 ≈ 0``).
   - 6 kolom bernama tanpa sinyal segmen pada pelanggan normal: ``kota_tier``,
     ``jumlah_alamat``, ``rating_beri_rata2``, serta 3 pendeteksi khusus bot
     (``durasi_sesi_menit``, ``klik_iklan_30h``, ``wishlist``).
3. Memangkas 4 fitur redundan/deterministik (``belanja_90h``, ``belanja_30h``,
   ``frekuensi_30h``, ``rasio_konversi``), 7 kategori berprofil datar, serta indikator
   ``is_web`` yang identik secara perilaku dengan ``android``.
4. Menerapkan pembobotan blok seimbang pada 26 fitur terpilih (13 perilaku inti w=1,0;
   3 CLR waktu belanja w=1,0; ``is_ios`` w=0,50; 9 CLR kategori informatif w=0,45)
   serta proyeksi PCA-12 untuk pemodelan kovarians penuh yang stabil.
5. Membuktikan dampak seleksi & pembobotan fitur lewat ablasi 5 konfigurasi pada K=6.

Jalankan langsung:
    python kode/src/clustering/features.py --run-ablation --save-json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.clustering.preprocess import (  # noqa: E402
    ANONYMOUS_N_COLUMNS,
    BOT_INTEGRITY_COLUMNS,
    ID_COLUMN,
    INFORMATIVE_KATEGORI_COLUMNS,
    LOW_SIGNAL_KATEGORI_COLUMNS,
    NOISE_NAMED_COLUMNS,
    PORSI_JAM_COLUMNS,
    PORSI_KATEGORI_COLUMNS,
    REDUNDANT_COLUMNS,
    deteksi_anomali_bot_aktivitas,
    muat_data_clustering,
    praproses_fitur_kanonik,
)
from src.config import FIGURES_DIR, RESULTS_DIR, SEED, siapkan_direktori  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    atur_gaya_grafik,
    label_batang,
    miring,
    tulis_json,
    atur_seed_global,
    sumbu_koma,
    catat_waktu,
)

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


def bangun_fitur_clustering(
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


def saring_semua_fitur(df_normal: pd.DataFrame, seed: int = SEED) -> list[dict[str, Any]]:
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


def _eval_clustering_config(
    X: np.ndarray,
    k: int = 6,
    gmm_reg: float = 1e-2,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluasi satu konfigurasi matriks fitur pada K=6 dengan K-Means dan GMM."""
    km1 = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    km2 = KMeans(n_clusters=k, n_init=10, random_state=seed + 57).fit(X)
    g1 = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=gmm_reg,
        n_init=2,
        max_iter=200,
        random_state=seed,
    ).fit(X)
    g2 = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=gmm_reg,
        n_init=2,
        max_iter=200,
        random_state=seed + 57,
    ).fit(X)
    l_g1 = g1.predict(X)
    l_g2 = g2.predict(X)
    sizes_pct = sorted(
        (pd.Series(l_g1).value_counts(normalize=True) * 100.0).round(1).tolist(),
        reverse=True,
    )
    return {
        "n_dims": int(X.shape[1]),
        "km_silhouette": float(silhouette_score(X, km1.labels_, sample_size=10_000, random_state=seed)),
        "gmm_silhouette": float(silhouette_score(X, l_g1, sample_size=10_000, random_state=seed)),
        "km_calinski_harabasz": float(calinski_harabasz_score(X, km1.labels_)),
        "km_davies_bouldin": float(davies_bouldin_score(X, km1.labels_)),
        "km_gmm_agreement_ari": float(adjusted_rand_score(km1.labels_, l_g1)),
        "km_seed_stability_ari": float(adjusted_rand_score(km1.labels_, km2.labels_)),
        "gmm_seed_stability_ari": float(adjusted_rand_score(l_g1, l_g2)),
        "gmm_mean_max_posterior": float(g1.predict_proba(X).max(axis=1).mean()),
        "gmm_sizes_pct": sizes_pct,
    }


def uji_ablasi_fitur(
    df_normal: pd.DataFrame,
    feat_bundle: dict[str, Any],
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """Bandingkan 5 konfigurasi seleksi & pembobotan fitur pada subsampel 30.000 pelanggan non-bot."""
    rng = np.random.default_rng(seed)
    idx_sub = rng.choice(len(df_normal), size=30_000, replace=False)
    sub_raw = df_normal.iloc[idx_sub]
    df_canon_sub = feat_bundle["df_canon"].iloc[idx_sub]

    # 1. All 59 raw (naif: termasuk n01-n15, 6 named noise, 4 redundant, fillna(0), OHE perangkat)
    raw_df = sub_raw.drop(columns=[ID_COLUMN]).copy()
    raw_df["rata2_keranjang"] = raw_df["rata2_keranjang"].fillna(0.0)
    raw_df = pd.get_dummies(raw_df, columns=["perangkat"], drop_first=False, dtype=np.float64)
    X_all_59 = StandardScaler().fit_transform(raw_df.to_numpy(dtype=np.float64))

    # 2. Drop n01-n15 saja (masih simpan 6 named noise, 4 redundant, OHE perangkat, 16 raw kat)
    no_n_df = raw_df.drop(columns=list(ANONYMOUS_N_COLUMNS))
    X_drop_n = StandardScaler().fit_transform(no_n_df.to_numpy(dtype=np.float64))

    # 3. 26D terpilih tanpa pembobotan blok (seluruh 26 fitur bobot 1,0)
    X_26_unw = StandardScaler().fit_transform(
        df_canon_sub[list(SELECTED_26_COLS)].to_numpy(dtype=np.float64)
    )

    # 4. 26D terpilih + pembobotan blok seimbang
    X_26_w = feat_bundle["X_weighted_26d"][idx_sub]

    # 5. Proyeksi PCA-12 dari 26D terbobot
    Z_pca12 = feat_bundle["Z_pca"][idx_sub]

    configs = [
        ("all_59_raw", X_all_59, 1e-2),
        ("drop_n15_only", X_drop_n, 1e-2),
        ("selected_26d_unweighted", X_26_unw, 1e-2),
        ("selected_26d_weighted", X_26_w, 1e-2),
        ("selected_pca12", Z_pca12, 1e-3),
    ]
    results: dict[str, dict[str, Any]] = {}
    for name, mat, reg in configs:
        results[name] = _eval_clustering_config(mat, k=6, gmm_reg=reg, seed=seed)
    return results


def buat_grafik_seleksi_fitur(
    screening: list[dict[str, Any]],
    ablation: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Buat grafik 2-panel Tugas 3b: penyaringan fitur (termasuk jebakan n06/n13) & ablasi."""
    atur_gaya_grafik()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 3.05), constrained_layout=True)

    # Panel A: OOF Ridge R^2 vs Sarle Bimodality Coefficient (mengungkap jebakan n06 & n13!)
    ax = axes[0]
    group_styles = {
        "selected_core_behavior": ("#2B6CB0", "o", "Inti & waktu terpilih"),
        "selected_time_clr": ("#2B6CB0", "o", None),
        "selected_device_ios": ("#2B6CB0", "o", None),
        "selected_category_clr": ("#2F855A", "s", "Kategori informatif (9)"),
        "dropped_low_signal_category": ("#68D391", "^", "Kategori datar (7)"),
        "dropped_redundant": ("#D69E2E", "D", "Redundan/deterministik (4)"),
        "dropped_bot_only_detector": ("#DD6B20", "P", "Khusus bot / noise bernama (6)"),
        "dropped_named_noise": ("#DD6B20", "P", None),
        "dropped_anonymous_noise": ("#C53030", "X", "Noise anonim n01–n15 (15)"),
    }
    seen_labels: set[str] = set()
    for row in screening:
        color, marker, lbl = group_styles[row["status"]]
        show_lbl = lbl if (lbl is not None and lbl not in seen_labels) else None
        if show_lbl is not None:
            seen_labels.add(show_lbl)
        ax.scatter(
            row["sarle_bimodality"],
            row["oof_ridge_r2"],
            color=color,
            marker=marker,
            s=28,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            label=show_lbl,
            zorder=3,
        )

    # Anotasi jebakan bimodal n06 & n13 serta batas bimodalitas Sarle
    ax.axvline(0.555, color="#718096", linestyle="--", linewidth=0.8)
    ax.annotate(
        "Jebakan bimodal\nn06 & n13\n(BC=0,67; R²≈0)",
        xy=(0.666, 0.005),
        xytext=(0.645, 0.28),
        fontsize=5.8,
        color="#9B2C2C",
        arrowprops={"arrowstyle": "->", "color": "#9B2C2C", "lw": 0.7},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#FFF5F5", "edgecolor": "#FEB2B2", "lw": 0.5},
    )
    ax.text(0.545, 0.99, "Batas bimodal\nSarle (0,555)", fontsize=5.6, color="#4A5568", va="top", ha="right")

    ax.set_xlabel("Koefisien bimodalitas Sarle (univariat)")
    ax.set_ylabel(f"Prediktabilitas silang OOF {miring('Ridge')} $R^2$")
    ax.set_title("(a) Penyaringan 59 Fitur: Sinyal vs Jebakan Noise")
    ax.set_xlim(0.28, 0.88)
    ax.set_ylim(-0.04, 1.05)
    sumbu_koma(ax, "xy")
    ax.legend(loc="upper right", fontsize=5.5, handletextpad=0.3, borderpad=0.25)

    # Panel B: Ablasi 5 konfigurasi fitur pada K=6 (Silhouette KM & Kesepakatan KM-GMM ARI)
    ax = axes[1]
    order = [
        ("all_59_raw", "Semua 59\nmentah"),
        ("drop_n15_only", "Buang\nn01–n15"),
        ("selected_26d_unweighted", "26D bersih\ntanpa bobot"),
        ("selected_26d_weighted", "26D bersih\nterbobot"),
        ("selected_pca12", "PCA-12\n(terpilih)"),
    ]
    x_pos = np.arange(len(order))
    sil_vals = [ablation[k]["km_silhouette"] for k, _ in order]
    ari_vals = [ablation[k]["km_gmm_agreement_ari"] for k, _ in order]
    w = 0.37

    b1 = ax.bar(x_pos - w / 2, sil_vals, width=w, color="#2B6CB0", label=f"{miring('Silhouette')} K-Means")
    b2 = ax.bar(x_pos + w / 2, ari_vals, width=w, color="#2F855A", label="Kesepakatan KM–GMM (ARI)")
    label_batang(ax, b1, desimal=3, ukuran=5.6)
    label_batang(ax, b2, desimal=2, ukuran=5.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([lbl for _, lbl in order], fontsize=6.2)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Skor evaluasi (K = 6)")
    ax.set_title("(b) Dampak Seleksi & Pembobotan Fitur pada K = 6")
    sumbu_koma(ax, "y")
    ax.legend(loc="upper left", fontsize=6.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 3b: Feature screening, weighting & ablation.")
    parser.add_argument("--run-ablation", action="store_true", help="Run 5-config feature ablation at K=6.")
    parser.add_argument("--save-json", action="store_true", help="Save compact JSON report to kode/results/.")
    args = parser.parse_args()

    atur_seed_global(SEED)
    siapkan_direktori()

    with catat_waktu("Task 3b: Feature Selection & Block Weighting"):
        df = muat_data_clustering()
        bundle = bangun_fitur_clustering(df, n_pca_components=DEFAULT_N_PCA, seed=SEED)
        screening = saring_semua_fitur(bundle["df_normal"], seed=SEED)

        ablation: dict[str, dict[str, Any]] = {}
        if args.run_ablation:
            ablation = uji_ablasi_fitur(bundle["df_normal"], bundle, seed=SEED)

        # Ringkas statistik penyaringan per kategori status
        by_status: dict[str, list[dict[str, Any]]] = {}
        for row in screening:
            by_status.setdefault(row["status"], []).append(row)

        status_summary: dict[str, dict[str, Any]] = {}
        for st, items in sorted(by_status.items()):
            status_summary[st] = {
                "count": len(items),
                "max_oof_r2": max(r["oof_ridge_r2"] for r in items),
                "max_spearman_sum_r2": max(r["spearman_sum_r2"] for r in items),
                "max_sarle_bimodality": max(r["sarle_bimodality"] for r in items),
                "features": [r["feature"] for r in items],
            }

        # Soroti jebakan bimodal n06 & n13 secara eksplisit
        bimodal_trap = [r for r in screening if r["feature"] in ("n06", "n13")]

        report: dict[str, Any] = {
            "n_total_raw_features": len(screening),
            "n_selected_transformed_dims": len(SELECTED_26_COLS),
            "n_pca_dims": DEFAULT_N_PCA,
            "pca_cumulative_variance_ratio": bundle["pca_cumulative_variance_ratio"],
            "pca_explained_variance_ratio": bundle["pca_explained_variance_ratio"],
            "block_weights": bundle["block_weights"],
            "block_variance_share_pct": bundle["block_variance_share_pct"],
            "bot_only_detectors_kept_for_3g": list(BOT_INTEGRITY_COLUMNS[1:]),
            "status_summary": status_summary,
            "bimodal_noise_trap": bimodal_trap,
            "screening_table": screening,
            "ablation_k6": ablation,
        }

        if ablation:
            fig_path = FIGURES_DIR / "task3_3b_feature_selection.png"
            buat_grafik_seleksi_fitur(screening, ablation, fig_path)
            print(f"Figur         : {fig_path.name}")

        if args.tulis_json:
            tulis_json(report, RESULTS_DIR / "task3_features.json")

        print(
            f"Seleksi fitur : 59 kandidat -> 26D terbobot -> PCA-{DEFAULT_N_PCA} "
            f"(varians kumulatif {bundle['pca_cumulative_variance_ratio']*100:.1f}%)"
        )
        print("Pangsa varians blok 26D:", {k: round(v, 1) for k, v in bundle["block_variance_share_pct"].items()})
        if ablation:
            print("Ablasi K=6:")
            for cfg, m in ablation.items():
                print(
                    f"  {cfg:<24s} ({m['n_dims']:2d}D): KM_sil={m['km_silhouette']:.4f}, "
                    f"GMM_sil={m['gmm_silhouette']:.4f}, KM-GMM_ARI={m['km_gmm_agreement_ari']:.4f}, "
                    f"GMM_stab={m['gmm_seed_stability_ari']:.4f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
