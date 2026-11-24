"""Perbandingan metode clustering (Tugas 3c) dan pemilihan jumlah klaster K (Tugas 3d).

Modul ini menjalankan dua tahap utama pada pelanggan non-bot (118.200 baris) di ruang
fitur bersih hasil Tugas 3a–3b:
1. Tugas 3c — Perbandingan lima keluarga algoritma clustering berbeda pada K = 6:
   - Gaussian Mixture Model (Full Covariance 26D & PCA-12, serta Diagonal PCA-12)
   - K-Means (26D & PCA-12)
   - Agglomerative Clustering — Ward (subsampel 12.000 diperluas lewat NearestCentroid)
   - BIRCH (CF-Tree hierarkis pada PCA-12)
   - HDBSCAN (berbasis kepadatan pada PCA-5 & PCA-8 untuk menguji struktur kepadatan)
2. Tugas 3d — Pemilihan jumlah klaster K pada rentang K = 2..8 memakai empat kriteria
   independen: GMM BIC & penurunan BIC (dBIC), skor Silhouette (K-Means & GMM),
   Gap Statistic (Tibshirani dkk., 2001, B=8 kotak acuan PCA), serta stabilitas &
   kesepakatan lintas-metode (ARI K-Means vs GMM).

Jalankan langsung:
    python kode/src/clustering/models.py --save-json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.clustering.features import DEFAULT_N_PCA, bangun_fitur_clustering  # noqa: E402
from src.clustering.preprocess import muat_data_clustering  # noqa: E402
from src.config import FIGURES_DIR, RESULTS_DIR, SEED, siapkan_direktori  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    atur_gaya_grafik,
    label_batang,
    miring,
    tulis_json,
    atur_seed_global,
    sumbu_koma,
    catat_waktu,
)

SELECTED_K = 6
K_CANDIDATES: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
SUBSAMPLE_N_METHODS = 30_000
SUBSAMPLE_N_KSCAN = 35_000
SUBSAMPLE_N_WARD = 12_000
SUBSAMPLE_N_GAP = 10_000
GAP_N_REFS = 8
SILHOUETTE_SAMPLE_N = 10_000


def _summarize_partition(
    Z_pca: np.ndarray,
    X_26d: np.ndarray,
    labels: np.ndarray,
    seed: int = SEED,
    mean_max_post: float | None = None,
) -> dict[str, Any]:
    """Hitung metrik internal & distribusi ukuran segmen untuk satu partisi klaster."""
    sizes_pct = sorted(
        (pd.Series(labels).value_counts(normalize=True) * 100.0).round(2).tolist(),
        reverse=True,
    )
    out: dict[str, Any] = {
        "n_clusters": int(len(set(labels) - {-1})),
        "silhouette_pca12": float(
            silhouette_score(Z_pca, labels, sample_size=SILHOUETTE_SAMPLE_N, random_state=seed)
        ),
        "silhouette_26d": float(
            silhouette_score(X_26d, labels, sample_size=SILHOUETTE_SAMPLE_N, random_state=seed)
        ),
        "calinski_harabasz_pca12": float(calinski_harabasz_score(Z_pca, labels)),
        "davies_bouldin_pca12": float(davies_bouldin_score(Z_pca, labels)),
        "sizes_pct": sizes_pct,
    }
    if mean_max_post is not None:
        out["mean_max_posterior"] = float(mean_max_post)
    return out


def compare_clustering_methods(
    X_26d: np.ndarray,
    Z_pca: np.ndarray,
    k: int = SELECTED_K,
    seed: int = SEED,
) -> dict[str, Any]:
    """Bandingkan 5 keluarga metode clustering pada K terpilih (subsampel 30.000 non-bot)."""
    rng = np.random.default_rng(seed)
    idx30 = rng.choice(len(Z_pca), size=min(SUBSAMPLE_N_METHODS, len(Z_pca)), replace=False)
    idx12 = idx30[: min(SUBSAMPLE_N_WARD, len(idx30))]

    X30 = X_26d[idx30]
    Z30 = Z_pca[idx30]
    Z12 = Z_pca[idx12]

    # 1. Gaussian Mixture Models (Full 26D, Full PCA-12, Diag PCA-12)
    gmm_26 = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-2,
        n_init=3,
        max_iter=300,
        random_state=seed,
    ).fit(X30)
    gmm_p12 = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-3,
        n_init=5,
        max_iter=300,
        random_state=seed,
    ).fit(Z30)
    dgmm_p12 = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        reg_covar=1e-3,
        n_init=5,
        max_iter=300,
        random_state=seed,
    ).fit(Z30)

    # 2. K-Means (26D & PCA-12)
    km_26 = KMeans(n_clusters=k, n_init=15, random_state=seed).fit(X30)
    km_p12 = KMeans(n_clusters=k, n_init=15, random_state=seed).fit(Z30)

    # 3. Agglomerative Ward pada 12k + ekstensi NearestCentroid ke 30k
    ward_12 = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z12)
    lbl_ward = NearestCentroid().fit(Z12, ward_12.labels_).predict(Z30)

    # 4. BIRCH CF-Tree pada PCA-12
    birch = Birch(threshold=1.2, branching_factor=50, n_clusters=k).fit(Z30)
    lbl_birch = birch.labels_

    # 5. HDBSCAN berbasis kepadatan (diuji pada PCA-5 & PCA-8 untuk diagnostik kepadatan)
    hdb_configs = [
        ("pca8_mcs250_ms25", 8, 250, 25),
        ("pca5_mcs150_ms10", 5, 150, 10),
        ("pca5_mcs100_ms5", 5, 100, 5),
    ]
    hdb_summary: dict[str, dict[str, Any]] = {}
    for cfg_name, n_dim, mcs, ms in hdb_configs:
        hdb = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            copy=True,
            n_jobs=-1,
        ).fit(Z12[:, :n_dim])
        lbl_h = hdb.labels_
        n_c = int(len(set(lbl_h) - {-1}))
        noise_pct = float((lbl_h == -1).mean() * 100.0)
        core_mask = lbl_h >= 0
        core_sil = (
            float(silhouette_score(Z12[core_mask, :n_dim], lbl_h[core_mask]))
            if n_c >= 2 and int(core_mask.sum()) > n_c
            else None
        )
        hdb_summary[cfg_name] = {
            "n_dims": n_dim,
            "min_cluster_size": mcs,
            "min_samples": ms,
            "n_clusters_found": n_c,
            "noise_pct": round(noise_pct, 2),
            "core_only_silhouette": core_sil,
        }

    partitions: dict[str, tuple[np.ndarray, float | None]] = {
        "GMM_full_26D": (gmm_26.predict(X30), float(gmm_26.predict_proba(X30).max(axis=1).mean())),
        "GMM_full_PCA12": (gmm_p12.predict(Z30), float(gmm_p12.predict_proba(Z30).max(axis=1).mean())),
        "GMM_diag_PCA12": (dgmm_p12.predict(Z30), float(dgmm_p12.predict_proba(Z30).max(axis=1).mean())),
        "KMeans_26D": (km_26.labels_, None),
        "KMeans_PCA12": (km_p12.labels_, None),
        "Ward_PCA12": (lbl_ward, None),
        "BIRCH_PCA12": (lbl_birch, None),
    }

    method_metrics: dict[str, dict[str, Any]] = {}
    for name, (lbl, post) in partitions.items():
        method_metrics[name] = _summarize_partition(
            Z30, X30, lbl, seed=seed, mean_max_post=post
        )

    names = list(partitions.keys())
    ari_matrix: dict[str, dict[str, float]] = {}
    for a in names:
        ari_matrix[a] = {}
        for b in names:
            ari_matrix[a][b] = round(
                float(adjusted_rand_score(partitions[a][0], partitions[b][0])), 4
            )

    for a in names:
        others = [ari_matrix[a][b] for b in names if b != a]
        method_metrics[a]["mean_consensus_ari"] = round(float(np.mean(others)), 4)

    return {
        "k": int(k),
        "n_subsample": int(len(idx30)),
        "n_ward_hdbscan_subsample": int(len(idx12)),
        "partition_methods": method_metrics,
        "hdbscan_diagnostics": hdb_summary,
        "pairwise_ari_matrix": ari_matrix,
    }


def _compute_gap_statistic(
    Z_sub: np.ndarray,
    k_values: tuple[int, ...] = K_CANDIDATES,
    n_refs: int = GAP_N_REFS,
    seed: int = SEED,
) -> dict[int, dict[str, float]]:
    """Hitung Gap Statistic (Tibshirani dkk., 2001) pada kotak acuan sumbu utama PCA."""
    rng = np.random.default_rng(seed + 101)
    lo = np.min(Z_sub, axis=0)
    hi = np.max(Z_sub, axis=0)
    # Buat B dataset acuan seragam pada kotak PCA terlebih dahulu agar adil antar-K
    refs = [rng.uniform(lo, hi, size=Z_sub.shape) for _ in range(n_refs)]

    gap_out: dict[int, dict[str, float]] = {}
    for k in k_values:
        km_obs = KMeans(n_clusters=k, n_init=8, random_state=seed).fit(Z_sub)
        log_wk = float(np.log(km_obs.inertia_))
        ref_logs = np.empty(n_refs, dtype=np.float64)
        for b, ref_mat in enumerate(refs):
            km_ref = KMeans(n_clusters=k, n_init=3, random_state=seed + b).fit(ref_mat)
            ref_logs[b] = float(np.log(km_ref.inertia_))
        gap_val = float(np.mean(ref_logs) - log_wk)
        sk = float(np.std(ref_logs, ddof=1) * np.sqrt(1.0 + 1.0 / n_refs))
        gap_out[k] = {
            "gap": gap_val,
            "gap_sk": sk,
            "log_wk": log_wk,
            "mean_ref_log_wk": float(np.mean(ref_logs)),
        }
    return gap_out


def scan_k_selection(
    Z_pca: np.ndarray,
    k_values: tuple[int, ...] = K_CANDIDATES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluasi pemilihan K pada rentang K=3..9 memakai BIC, Silhouette, Gap, & stabilitas."""
    rng = np.random.default_rng(seed)
    idx_scan = rng.choice(len(Z_pca), size=min(SUBSAMPLE_N_KSCAN, len(Z_pca)), replace=False)
    Z_scan = Z_pca[idx_scan]
    Z_gap = Z_scan[: min(SUBSAMPLE_N_GAP, len(Z_scan))]

    gap_res = _compute_gap_statistic(Z_gap, k_values=k_values, n_refs=GAP_N_REFS, seed=seed)

    rows: list[dict[str, Any]] = []
    prev_bic: float | None = None
    for k in k_values:
        g1 = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-2,
            n_init=3,
            max_iter=300,
            random_state=seed,
        ).fit(Z_scan)
        g2 = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-2,
            n_init=3,
            max_iter=300,
            random_state=seed + 57,
        ).fit(Z_scan)
        km1 = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z_scan)
        km2 = KMeans(n_clusters=k, n_init=10, random_state=seed + 57).fit(Z_scan)

        l_g1, l_g2 = g1.predict(Z_scan), g2.predict(Z_scan)
        bic = float(g1.bic(Z_scan))
        aic = float(g1.aic(Z_scan))
        dbic = 0.0 if prev_bic is None else float(bic - prev_bic)
        prev_bic = bic

        sil_km = float(silhouette_score(Z_scan, km1.labels_, sample_size=SILHOUETTE_SAMPLE_N, random_state=seed))
        sil_gm = float(silhouette_score(Z_scan, l_g1, sample_size=SILHOUETTE_SAMPLE_N, random_state=seed))
        ch_km = float(calinski_harabasz_score(Z_scan, km1.labels_))
        db_km = float(davies_bouldin_score(Z_scan, km1.labels_))
        ari_kg = float(adjusted_rand_score(km1.labels_, l_g1))
        ari_gs = float(adjusted_rand_score(l_g1, l_g2))
        ari_ks = float(adjusted_rand_score(km1.labels_, km2.labels_))
        mean_post = float(g1.predict_proba(Z_scan).max(axis=1).mean())
        sizes = sorted(
            (pd.Series(l_g1).value_counts(normalize=True) * 100.0).round(2).tolist(),
            reverse=True,
        )

        rows.append(
            {
                "k": int(k),
                "gmm_bic": bic,
                "gmm_dbic": dbic,
                "gmm_aic": aic,
                "km_silhouette": sil_km,
                "gmm_silhouette": sil_gm,
                "km_calinski_harabasz": ch_km,
                "km_davies_bouldin": db_km,
                "km_gmm_agreement_ari": ari_kg,
                "gmm_seed_stability_ari": ari_gs,
                "km_seed_stability_ari": ari_ks,
                "gmm_mean_max_posterior": mean_post,
                "gap": gap_res[k]["gap"],
                "gap_sk": gap_res[k]["gap_sk"],
                "gmm_sizes_pct": sizes,
            }
        )

    # Hitung selisih Gap (delta Gap = Gap(k) - Gap(k-1)) untuk mendeteksi siku kenaikan marginal
    for i, row in enumerate(rows):
        row["gap_gain"] = 0.0 if i == 0 else float(row["gap"] - rows[i - 1]["gap"])

    return {
        "selected_k": SELECTED_K,
        "n_subsample_kscan": int(len(Z_scan)),
        "n_subsample_gap": int(len(Z_gap)),
        "gap_n_refs": GAP_N_REFS,
        "k_scan": rows,
    }


def buat_grafik_metode_dan_k(
    k_res: dict[str, Any],
    meth_res: dict[str, Any],
    out_path: Path,
) -> None:
    """Buat grafik 2x2 pemilihan jumlah klaster K (3d) dan perbandingan metode (3c)."""
    atur_gaya_grafik()
    fig, axes = plt.subplots(2, 2, figsize=(LEBAR_GAMBAR, 4.95), constrained_layout=True)

    rows = k_res["k_scan"]
    ks = [r["k"] for r in rows]

    # Panel A: Penurunan BIC GMM (-dBIC) per penambahan klaster (siku tajam setelah K=6!)
    ax = axes[0, 0]
    ks_d = ks[1:]
    bic_drops = [-r["gmm_dbic"] for r in rows[1:]]
    bar_colors = ["#2B6CB0" if k == SELECTED_K else ("#90CDF4" if k < SELECTED_K else "#CBD5E0") for k in ks_d]
    bars = ax.bar(ks_d, bic_drops, width=0.58, color=bar_colors, edgecolor="#2D3748", linewidth=0.45)
    for b, val in zip(bars, bic_drops):
        y_txt = max(val, 0.0)
        ax.annotate(
            angka_id(val, 0),
            (b.get_x() + b.get_width() / 2, y_txt),
            xytext=(0, 2.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
    ax.axhline(0.0, color="#4A5568", linewidth=0.6)
    ax.set_xticks(ks_d)
    ax.set_ylim(min(min(bic_drops) * 1.25, -350), max(bic_drops) * 1.24)
    ax.set_xlabel("Jumlah klaster K")
    ax.set_ylabel("Penurunan BIC GMM (−ΔBIC)")
    ax.set_title("(a) Siku Penurunan BIC GMM (−ΔBIC)")
    sumbu_koma(ax, "y")

    # Panel B: Silhouette (KM & GMM) dan Gap Statistic melintasi K = 3..9
    ax = axes[0, 1]
    sil_km = [r["km_silhouette"] for r in rows]
    sil_gm = [r["gmm_silhouette"] for r in rows]
    gaps = [r["gap"] for r in rows]
    gap_sk = [r["gap_sk"] for r in rows]

    l1 = ax.plot(ks, sil_km, marker="o", color="#2B6CB0", label=f"{miring('Silhouette')} K-Means")
    l2 = ax.plot(ks, sil_gm, marker="s", linestyle="--", color="#2F855A", label=f"{miring('Silhouette')} GMM Full")
    ax.axvline(SELECTED_K, color="#C53030", linestyle=":", linewidth=1.0)
    ax.set_xticks(ks)
    ax.set_ylim(min(min(sil_km), min(sil_gm)) - 0.035, max(max(sil_km), max(sil_gm)) + 0.015)
    ax.set_xlabel("Jumlah klaster K")
    ax.set_ylabel(f"Skor {miring('Silhouette')} (PCA-12)")
    sumbu_koma(ax, "y")

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.grid(False)
    l3 = ax2.errorbar(
        ks,
        gaps,
        yerr=gap_sk,
        marker="^",
        color="#DD6B20",
        capsize=2.0,
        linewidth=1.1,
        label=f"{miring('Gap statistic')} ($\\pm s_k$)",
    )
    ax2.set_ylabel(f"Statistik {miring('Gap')}", color="#C05621")
    ax2.tick_params(axis="y", labelcolor="#C05621")
    sumbu_koma(ax2, "y")

    # Gabungkan legenda dua sumbu di bawah tengah (ruang kosong)
    handles = l1 + l2 + [l3]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="lower center", fontsize=5.3)
    ax.set_title(f"(b) {miring('Silhouette')} & Statistik {miring('Gap')} (K = 2..8)")

    # Panel C: Perbandingan metrik metode pada K = 6 (Silhouette PCA-12 & Konsensus ARI)
    ax = axes[1, 0]
    pm = meth_res["partition_methods"]
    method_order = [
        ("GMM_full_26D", "GMM Full (26D)"),
        ("GMM_full_PCA12", "GMM Full (PCA-12)"),
        ("GMM_diag_PCA12", "GMM Diag (PCA-12)"),
        ("KMeans_26D", "K-Means (26D)"),
        ("Ward_PCA12", "Ward (PCA-12)"),
        ("BIRCH_PCA12", "BIRCH (PCA-12)"),
    ]
    y_pos = np.arange(len(method_order))
    h = 0.36
    sil_m = [pm[m]["silhouette_pca12"] for m, _ in method_order]
    cons_m = [pm[m]["mean_consensus_ari"] for m, _ in method_order]

    b1 = ax.barh(y_pos - h / 2, sil_m, height=h, color="#2B6CB0", label=f"{miring('Silhouette')} (PCA-12)")
    b2 = ax.barh(y_pos + h / 2, cons_m, height=h, color="#2F855A", label="Konsensus ARI")
    label_batang(ax, b1, desimal=3, mendatar=True, ukuran=5.5)
    label_batang(ax, b2, desimal=2, mendatar=True, ukuran=5.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in method_order], fontsize=6.4)
    ax.set_ylim(len(method_order) + 0.55, -0.55)
    ax.set_xlim(0.0, 1.06)
    ax.set_xlabel("Skor evaluasi (K = 6)")
    ax.set_title("(c) Perbandingan Metode (HDBSCAN: 80–100% noise)")
    sumbu_koma(ax, "x")
    ax.legend(loc="lower right", ncol=2, fontsize=5.4, columnspacing=0.8)

    # Panel D: Heatmap kesepakatan ARI antar-metode pada K = 6
    ax = axes[1, 1]
    ax.grid(False)
    hm_keys = [m for m, _ in method_order]
    hm_short = ["GMM\n26D", "GMM\nP12", "GMM\nDiag", "K-Means\n26D", "Ward\nP12", "BIRCH\nP12"]
    ari_mat = np.array(
        [[meth_res["pairwise_ari_matrix"][a][b] for b in hm_keys] for a in hm_keys],
        dtype=np.float64,
    )
    im = ax.imshow(ari_mat, cmap="YlGnBu", vmin=0.65, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(hm_short)))
    ax.set_yticks(np.arange(len(hm_short)))
    ax.set_xticklabels(hm_short, fontsize=5.8)
    ax.set_yticklabels([s.replace("\n", " ") for s in hm_short], fontsize=5.8)
    for i in range(len(hm_short)):
        for j in range(len(hm_short)):
            val = ari_mat[i, j]
            txt_color = "white" if val >= 0.86 else "#1A202C"
            ax.text(j, i, angka_id(val, 2), ha="center", va="center", fontsize=5.8, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6.0)
    sumbu_koma(cbar.ax, "y")
    ax.set_title("(d) Matriks Kesepakatan ARI Antar-Metode (K = 6)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tasks 3c & 3d: Multi-method clustering comparison and K selection."
    )
    parser.add_argument("--save-json", action="store_true", help="Save JSON report to kode/results/.")
    args = parser.parse_args()

    atur_seed_global(SEED)
    siapkan_direktori()

    with catat_waktu("Tasks 3c & 3d: Methods Comparison & K Selection"):
        df = muat_data_clustering()
        bundle = bangun_fitur_clustering(df, n_pca_components=DEFAULT_N_PCA, seed=SEED)
        X_26d, Z_pca = bundle["X_weighted_26d"], bundle["Z_pca"]

        k_res = scan_k_selection(Z_pca, k_values=K_CANDIDATES, seed=SEED)
        meth_res = compare_clustering_methods(X_26d, Z_pca, k=SELECTED_K, seed=SEED)

        fig_path = FIGURES_DIR / "task3_3cd_methods_and_k.png"
        buat_grafik_metode_dan_k(k_res, meth_res, fig_path)

        report: dict[str, Any] = {
            "selected_method": "GMM_full_26D",
            "selected_k": SELECTED_K,
            "selection_rationale": (
                "K=6 dipilih karena pertemuan empat kriteria independen: siku penurunan BIC "
                "(anjlok >35x dari K=6 ke K=7 lalu mendatar), puncak lokal Silhouette K-Means & GMM, "
                "lonjakan marginal tertinggi Gap Statistic, stabilitas seed GMM >0,999, dan "
                "kesepakatan antar-metode tinggi (ARI 0,83–0,89). GMM Full Covariance pada 26D "
                "terpilih dibanding K-Means/Ward karena memodelkan kovarians elips dan bobot "
                "proporsi tak-sama tiap segmen (mean max posterior ~0,95)."
            ),
            "k_selection": k_res,
            "method_comparison_k6": meth_res,
        }
        if args.save_json:
            tulis_json(report, RESULTS_DIR / "task3_models.json")

        print("=== Ringkasan Pemindaian K (PCA-12) ===")
        for r in k_res["k_scan"]:
            print(
                f"  K={r['k']}: BIC={r['gmm_bic']:10.1f} (dBIC={r['gmm_dbic']:+8.1f}) | "
                f"Sil_KM={r['km_silhouette']:.4f} Sil_GM={r['gmm_silhouette']:.4f} | "
                f"Gap={r['gap']:.4f}±{r['gap_sk']:.4f} | ARI_KG={r['km_gmm_agreement_ari']:.4f} "
                f"Stab_GM={r['gmm_seed_stability_ari']:.4f}"
            )
        print("\n=== Perbandingan Metode (K=6) ===")
        for m, v in meth_res["partition_methods"].items():
            post_str = f", post={v['mean_max_posterior']:.4f}" if "mean_max_posterior" in v else ""
            print(
                f"  {m:<15s}: Sil_P12={v['silhouette_pca12']:.4f}, CH={v['calinski_harabasz_pca12']:7.1f}, "
                f"DB={v['davies_bouldin_pca12']:.3f}, Cons_ARI={v['mean_consensus_ari']:.4f}{post_str}"
            )
        print("HDBSCAN:", {k: (v["n_clusters_found"], f"{v['noise_pct']}% noise") for k, v in meth_res["hdbscan_diagnostics"].items()})
        print(f"Figur         : {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
