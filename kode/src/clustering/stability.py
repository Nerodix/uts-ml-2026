"""Evaluasi stabilitas clustering lewat subsampling/bootstrap (Tugas 3e).

Modul ini menguji keandalan hasil segmentasi terhadap fluktuasi sampel melalui dua
skema subsampling 80% tanpa pengembalian:
1. Pemindaian stabilitas lintas jumlah klaster K in {4, 5, 6, 7, 8} dengan R = 12
   subsampel 80% independen (66 pasangan ARI pada irisan observasi bersama) untuk
   GMM Full Covariance (PCA-12) dan K-Means (PCA-12).
2. Uji stabilitas model final produksi (GMM Full Covariance 26D, K = 6) pada R = 15
   subsampel 80% dari 50.000 pelanggan non-bot (105 pasangan ARI) beserta stabilitas
   Jaccard per segmen (penyelarasan optimal via algoritma Hungarian).

Jalankan langsung:
    python kode/src/clustering/stability.py --save-json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

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
    miring,
    tulis_json,
    atur_seed_global,
    sumbu_koma,
    catat_waktu,
)

K_STABILITY_RANGE: tuple[int, ...] = (4, 5, 6, 7, 8)
N_POOL_KSCAN = 25_000
N_RUNS_KSCAN = 12
N_POOL_FINAL = 50_000
N_RUNS_FINAL = 15
SUBSAMPLE_FRAC = 0.80


def _pairwise_subsample_ari(
    run_indices: list[np.ndarray],
    run_labels: list[np.ndarray],
    n_pool: int,
) -> list[float]:
    """Hitung seluruh pasangan ARI antar-run pada irisan observasi yang sama-sama terpilih."""
    # Peta cepat: matriks (n_runs, n_pool) bernilai -1 bila tidak termasuk subsampel
    n_runs = len(run_indices)
    label_mat = np.full((n_runs, n_pool), -1, dtype=np.int16)
    for r, (idx, lbl) in enumerate(zip(run_indices, run_labels)):
        label_mat[r, idx] = lbl

    aris: list[float] = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            shared = (label_mat[i] >= 0) & (label_mat[j] >= 0)
            aris.append(float(adjusted_rand_score(label_mat[i, shared], label_mat[j, shared])))
    return aris


def _summarize_ari(aris: list[float]) -> dict[str, float]:
    arr = np.asarray(aris, dtype=np.float64)
    return {
        "n_pairs": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _name_clusters_deterministic(df_pool: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Petakan ID klaster mentah K=6 ke nama segmen deterministik berdasarkan ciri khas dominan."""
    tmp = df_pool.copy()
    tmp["_c"] = labels
    stats_by_c: dict[int, dict[str, float]] = {}
    for cid, grp in tmp.groupby("_c"):
        stats_by_c[int(cid)] = {
            "vch": float(grp["persen_voucher"].median()),
            "pyl": float(grp["persen_paylater"].median()),
            "rec": float(grp["recency_hari"].median()),
            "krj": float(grp["rata2_keranjang"].median()),
            "mbr": float(grp["lama_member_bulan"].median()),
            "pagi": float(grp["porsi_jam_pagi"].mean()),
        }

    remaining = set(stats_by_c.keys())
    mapping: dict[int, str] = {}

    def take_best(name: str, key: str, reverse: bool = True) -> None:
        chosen = sorted(remaining, key=lambda c: stats_by_c[c][key], reverse=reverse)[0]
        mapping[chosen] = name
        remaining.remove(chosen)

    take_best("Impulsif PayLater & Retur Tinggi", "pyl", reverse=True)
    take_best("Premium / Affluent iOS", "krj", reverse=True)
    take_best("Pasif / Dormant COD", "rec", reverse=True)
    take_best("Member Baru Window-Shopper", "mbr", reverse=False)
    take_best("Pemburu Promo & Flash Sale Malam", "vch", reverse=True)
    take_best("Keluarga Pembeli Rutin Pagi", "pagi", reverse=True)
    return mapping


def evaluasi_stabilitas_lintas_k(
    Z_pca: np.ndarray,
    k_values: tuple[int, ...] = K_STABILITY_RANGE,
    n_runs: int = N_RUNS_KSCAN,
    seed: int = SEED,
) -> dict[str, Any]:
    """Uji stabilitas subsampling 80% pada K=4..8 untuk GMM Full (PCA-12) dan K-Means (PCA-12)."""
    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(Z_pca), size=min(N_POOL_KSCAN, len(Z_pca)), replace=False)
    Z_pool = Z_pca[pool_idx]
    n_pool = len(Z_pool)
    sub_size = int(round(n_pool * SUBSAMPLE_FRAC))

    # Gunakan daftar subsampel identik lintas K dan lintas metode agar perbandingan 100% adil
    sub_indices = [
        rng.choice(n_pool, size=sub_size, replace=False) for _ in range(n_runs)
    ]

    gmm_by_k: dict[str, dict[str, Any]] = {}
    km_by_k: dict[str, dict[str, Any]] = {}

    for k in k_values:
        gmm_labels: list[np.ndarray] = []
        km_labels: list[np.ndarray] = []
        for r, idx in enumerate(sub_indices):
            Zs = Z_pool[idx]
            g = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-2,
                n_init=2,
                max_iter=200,
                random_state=seed + r * 17,
            ).fit(Zs)
            km = KMeans(n_clusters=k, n_init=10, random_state=seed + r * 17).fit(Zs)
            gmm_labels.append(g.predict(Zs))
            km_labels.append(km.labels_)

        gmm_aris = _pairwise_subsample_ari(sub_indices, gmm_labels, n_pool)
        km_aris = _pairwise_subsample_ari(sub_indices, km_labels, n_pool)
        gmm_by_k[str(k)] = _summarize_ari(gmm_aris)
        km_by_k[str(k)] = _summarize_ari(km_aris)

    return {
        "n_pool": n_pool,
        "subsample_size": sub_size,
        "subsample_frac": SUBSAMPLE_FRAC,
        "n_runs": n_runs,
        "gmm_full_pca12": gmm_by_k,
        "kmeans_pca12": km_by_k,
    }


def evaluasi_stabilitas_model_final(
    df_normal: pd.DataFrame,
    X_26d: np.ndarray,
    k: int = 6,
    n_runs: int = N_RUNS_FINAL,
    seed: int = SEED,
) -> dict[str, Any]:
    """Uji stabilitas model final (GMM Full 26D, K=6) pada 15 subsampel 80% + Jaccard per segmen."""
    rng = np.random.default_rng(seed + 77)
    pool_idx = rng.choice(len(X_26d), size=min(N_POOL_FINAL, len(X_26d)), replace=False)
    X_pool = X_26d[pool_idx]
    df_pool = df_normal.iloc[pool_idx]
    n_pool = len(X_pool)
    sub_size = int(round(n_pool * SUBSAMPLE_FRAC))

    # Fit referensi pada seluruh pool untuk penyelarasan Jaccard per segmen
    ref_gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-2,
        n_init=3,
        max_iter=300,
        random_state=seed,
    ).fit(X_pool)
    ref_labels = ref_gmm.predict(X_pool)
    cid_to_name = _name_clusters_deterministic(df_pool, ref_labels)

    sub_indices: list[np.ndarray] = []
    run_labels: list[np.ndarray] = []
    ari_vs_ref: list[float] = []
    jaccard_per_seg: dict[str, list[float]] = {name: [] for name in cid_to_name.values()}

    for r in range(n_runs):
        idx = rng.choice(n_pool, size=sub_size, replace=False)
        Xs = X_pool[idx]
        g = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-2,
            n_init=2,
            max_iter=250,
            random_state=seed + r * 31,
        ).fit(Xs)
        lbl = g.predict(Xs)
        ref_sub = ref_labels[idx]

        sub_indices.append(idx)
        run_labels.append(lbl)
        ari_vs_ref.append(float(adjusted_rand_score(ref_sub, lbl)))

        # Matriks Jaccard (k x k) antara label referensi & subsampel, selaraskan via Hungarian
        jac_mat = np.zeros((k, k), dtype=np.float64)
        for ci in range(k):
            m_ref = ref_sub == ci
            for cj in range(k):
                m_run = lbl == cj
                inter = int(np.sum(m_ref & m_run))
                union = int(np.sum(m_ref | m_run))
                jac_mat[ci, cj] = inter / union if union > 0 else 0.0
        row_ind, col_ind = linear_sum_assignment(-jac_mat)
        for ci, cj in zip(row_ind, col_ind):
            jaccard_per_seg[cid_to_name[int(ci)]].append(float(jac_mat[ci, cj]))

    pairwise_aris = _pairwise_subsample_ari(sub_indices, run_labels, n_pool)
    per_segment_summary: dict[str, dict[str, float]] = {}
    for seg_name, vals in jaccard_per_seg.items():
        arr = np.asarray(vals, dtype=np.float64)
        per_segment_summary[seg_name] = {
            "mean_jaccard": float(np.mean(arr)),
            "std_jaccard": float(np.std(arr, ddof=1)),
            "min_jaccard": float(np.min(arr)),
            "max_jaccard": float(np.max(arr)),
        }

    return {
        "model": "GMM_full_26D",
        "k": int(k),
        "n_pool": n_pool,
        "subsample_size": sub_size,
        "subsample_frac": SUBSAMPLE_FRAC,
        "n_runs": n_runs,
        "pairwise_ari": _summarize_ari(pairwise_aris),
        "ari_vs_reference": _summarize_ari(ari_vs_ref),
        "per_segment_jaccard": per_segment_summary,
    }


def buat_grafik_stabilitas(
    k_stab: dict[str, Any],
    final_stab: dict[str, Any],
    out_path: Path,
) -> None:
    """Buat grafik 2-panel Tugas 3e: stabilitas ARI lintas K dan Jaccard per segmen K=6."""
    atur_gaya_grafik()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.95), constrained_layout=True)

    # Panel A: Stabilitas pairwise ARI (mean ± SD terklip pada [0, 1]) lintas K = 4..8
    ax = axes[0]
    ks = [int(k) for k in k_stab["gmm_full_pca12"].keys()]
    x = np.arange(len(ks))
    g_mean = np.array([k_stab["gmm_full_pca12"][str(k)]["mean"] for k in ks], dtype=np.float64)
    g_std = np.array([k_stab["gmm_full_pca12"][str(k)]["std"] for k in ks], dtype=np.float64)
    k_mean = np.array([k_stab["kmeans_pca12"][str(k)]["mean"] for k in ks], dtype=np.float64)
    k_std = np.array([k_stab["kmeans_pca12"][str(k)]["std"] for k in ks], dtype=np.float64)

    g_yerr = np.vstack([np.minimum(g_std, g_mean), np.minimum(g_std, 1.0 - g_mean)])
    k_yerr = np.vstack([np.minimum(k_std, k_mean), np.minimum(k_std, 1.0 - k_mean)])

    ax.errorbar(
        x - 0.08,
        g_mean,
        yerr=g_yerr,
        marker="o",
        color="#2B6CB0",
        capsize=2.5,
        linewidth=1.3,
        label="GMM Full (PCA-12)",
    )
    ax.errorbar(
        x + 0.08,
        k_mean,
        yerr=k_yerr,
        marker="s",
        linestyle="--",
        color="#2F855A",
        capsize=2.5,
        linewidth=1.2,
        label="K-Means (PCA-12)",
    )
    # Tandai titik model final GMM Full 26D pada K=6
    k6_pos = ks.index(6)
    f_mean = final_stab["pairwise_ari"]["mean"]
    ax.scatter(
        [k6_pos],
        [f_mean],
        marker="*",
        s=75,
        color="#C53030",
        zorder=5,
        label=f"GMM Full 26D final ({angka_id(f_mean, 3)})",
    )
    ax.axvspan(k6_pos - 0.35, k6_pos + 0.35, color="#EBF8FF", alpha=0.75, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    y_lo = max(0.0, float(np.min(g_mean - g_yerr[0])) - 0.08)
    ax.set_ylim(min(y_lo, 0.55), 1.04)
    ax.set_ylabel(f"{miring('Pairwise')} ARI (subsampling 80%)")
    ax.set_xlabel("Jumlah klaster K")
    ax.set_title("(a) Stabilitas ARI Subsampling 80% Lintas K")
    sumbu_koma(ax, "y")
    ax.legend(loc="lower left", fontsize=5.6)

    # Panel B: Stabilitas Jaccard per segmen pada model final GMM Full 26D (K=6, 15 run)
    ax = axes[1]
    seg_items = sorted(
        final_stab["per_segment_jaccard"].items(),
        key=lambda kv: kv[1]["mean_jaccard"],
        reverse=True,
    )
    short_names = {
        "Pemburu Promo & Flash Sale Malam": "Pemburu Promo Malam",
        "Keluarga Pembeli Rutin Pagi": "Keluarga Rutin Pagi",
        "Pasif / Dormant COD": "Pasif / Dormant COD",
        "Premium / Affluent iOS": "Premium / Affluent iOS",
        "Impulsif PayLater & Retur Tinggi": "Impulsif PayLater",
        "Member Baru Window-Shopper": "Window-Shopper Baru",
    }
    labels = [short_names.get(k, k) for k, _ in seg_items]
    means = [v["mean_jaccard"] for _, v in seg_items]
    stds = [v["std_jaccard"] for _, v in seg_items]
    mins = [v["min_jaccard"] for _, v in seg_items]
    y_pos = np.arange(len(labels))

    bars = ax.barh(
        y_pos,
        means,
        xerr=stds,
        height=0.55,
        color="#2B6CB0",
        ecolor="#2D3748",
        capsize=2.0,
        alpha=0.90,
    )
    for b, m, s, mn in zip(bars, means, stds, mins):
        ax.annotate(
            f"{angka_id(m, 3)} (min {angka_id(mn, 3)})",
            (m + s, b.get_y() + b.get_height() / 2),
            xytext=(3.0, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.5,
        )
    ax.axvline(0.85, color="#718096", linestyle="--", linewidth=0.8)
    ax.text(0.855, -0.58, "Batas stabil (0,85)", ha="left", va="center", fontsize=5.5, color="#4A5568")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_ylim(len(labels) - 0.35, -0.85)
    ax.set_xlim(0.75, 1.15)
    ax.set_xlabel(f"Indeks {miring('Jaccard')} (15 subsampel 80%)")
    ax.set_title("(b) Stabilitas per Segmen (GMM Full 26D, K = 6)")
    sumbu_koma(ax, "x")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 3e: Subsampling stability evaluation.")
    parser.add_argument("--save-json", action="store_true", help="Save JSON report to kode/results/.")
    args = parser.parse_args()

    atur_seed_global(SEED)
    siapkan_direktori()

    with catat_waktu("Task 3e: Clustering Stability Evaluation"):
        df = muat_data_clustering()
        bundle = bangun_fitur_clustering(df, n_pca_components=DEFAULT_N_PCA, seed=SEED)

        k_stab = evaluasi_stabilitas_lintas_k(
            bundle["Z_pca"], k_values=K_STABILITY_RANGE, n_runs=N_RUNS_KSCAN, seed=SEED
        )
        final_stab = evaluasi_stabilitas_model_final(
            bundle["df_normal"], bundle["X_weighted_26d"], k=6, n_runs=N_RUNS_FINAL, seed=SEED
        )

        fig_path = FIGURES_DIR / "task3_3e_stability.png"
        buat_grafik_stabilitas(k_stab, final_stab, fig_path)

        report: dict[str, Any] = {
            "stability_across_k": k_stab,
            "final_model_stability": final_stab,
        }
        if args.save_json:
            tulis_json(report, RESULTS_DIR / "task3_stability.json")

        print("=== Stabilitas Subsampling 80% Lintas K (66 pasangan ARI) ===")
        for k in K_STABILITY_RANGE:
            gm = k_stab["gmm_full_pca12"][str(k)]
            km = k_stab["kmeans_pca12"][str(k)]
            print(
                f"  K={k}: GMM_ARI={gm['mean']:.4f}±{gm['std']:.4f} (min={gm['min']:.4f}) | "
                f"KM_ARI={km['mean']:.4f}±{km['std']:.4f} (min={km['min']:.4f})"
            )
        fp = final_stab["pairwise_ari"]
        print(
            f"\nModel final (GMM Full 26D, K=6, 105 pasangan): "
            f"Pairwise ARI={fp['mean']:.4f}±{fp['std']:.4f} (median={fp['median']:.4f}, min={fp['min']:.4f})"
        )
        print("Jaccard per segmen:")
        for seg, v in sorted(
            final_stab["per_segment_jaccard"].items(), key=lambda kv: kv[1]["mean_jaccard"], reverse=True
        ):
            print(f"  {seg:<35s}: {v['mean_jaccard']:.4f} ± {v['std_jaccard']:.4f} (min={v['min_jaccard']:.4f})")
        print(f"Figur         : {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
