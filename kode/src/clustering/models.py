"""Perbandingan metode clustering (Tugas 3c) dan pemilihan jumlah klaster K (Tugas 3d).

Bagian 3c membandingkan lima keluarga algoritma clustering berbeda pada pelanggan
non-bot (118.200 baris) pada ruang fitur bersih hasil Tugas 3a–3b:
1. Gaussian Mixture Model (Full Covariance 26D & PCA-12, serta Diagonal PCA-12)
2. K-Means (26D & PCA-12)
3. Agglomerative Clustering — Ward (subsampel 12.000 diperluas lewat NearestCentroid)
4. BIRCH (CF-Tree hierarkis pada PCA-12)
5. HDBSCAN (berbasis kepadatan pada PCA-5 & PCA-8 untuk menguji struktur kepadatan)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

from src.config import SEED  # noqa: E402

SELECTED_K = 6
SUBSAMPLE_N_METHODS = 30_000
SUBSAMPLE_N_WARD = 12_000
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

    # Hitung rata-rata konsensus ARI tiap metode terhadap 6 metode lainnya
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
