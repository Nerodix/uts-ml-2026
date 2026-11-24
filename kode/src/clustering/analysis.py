"""Profil segmen pelanggan, rekomendasi aksi pemasaran (Tugas 3f), dan deteksi anomali (Tugas 3g).

Bagian 3f melatih model produksi akhir (Gaussian Mixture Model Full Covariance pada 26D
terbobot, K = 6) atas seluruh 118.200 pelanggan non-bot, memetakan ID segmen secara
deterministik ke 0..5, merangkum profil perilaku dan komposisi tiap segmen, serta
merumuskan satu aksi pemasaran terukur untuk setiap segmen.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.clustering.preprocess import (  # noqa: E402
    PORSI_JAM_COLUMNS,
    PORSI_KATEGORI_COLUMNS,
)
from src.config import SEED  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    configure_plotting,
    miring,
    sumbu_koma,
)

FINAL_K = 6

# Metadata segmen tetap (ID 0..5 diselaraskan secara deterministik berdasarkan ciri dominan)
SEGMENT_CATALOG: dict[int, dict[str, str]] = {
    0: {
        "name": "Pemburu Promo & Flash Sale Malam",
        "short_name": "Pemburu Promo Malam",
        "color": "#DD6B20",
        "marketing_action": (
            "Kirim notifikasi push flash sale malam (20.00–23.00) dengan voucher bertingkat "
            "berbasis minimum transaksi (mis. Rp 150 ribu) untuk menaikkan nilai keranjang "
            "pada kategori fashion, kecantikan, dan voucher digital tanpa menggerus margin."
        ),
    },
    1: {
        "name": "Keluarga Pembeli Rutin Pagi",
        "short_name": "Keluarga Rutin Pagi",
        "color": "#2F855A",
        "marketing_action": (
            "Tawarkan langganan isi ulang otomatis (auto-replenish) pagi hari untuk paket "
            "makanan/minuman, kebutuhan rumah tangga, dan ibu & bayi disertai poin loyalti "
            "serta gratis ongkir berjenjang."
        ),
    },
    2: {
        "name": "Pelanggan Pasif / Dormant COD",
        "short_name": "Pasif / Dormant COD",
        "color": "#718096",
        "marketing_action": (
            "Jalankan kampanye win-back lewat WhatsApp/email dengan voucher gratis ongkir COD "
            "bernilai terjangkau dan masa berlaku 7 hari untuk memicu kembali transaksi "
            "pertama pelanggan berisiko churn."
        ),
    },
    3: {
        "name": "Pelanggan Premium / Affluent iOS",
        "short_name": "Premium / Affluent iOS",
        "color": "#2B6CB0",
        "marketing_action": (
            "Berikan keanggotaan VIP non-diskon: prioritas pengiriman same-day, perpanjangan "
            "garansi resmi, dan akses awal (early access) peluncuran elektronik, perabot, "
            "serta gadget premium."
        ),
    },
    4: {
        "name": "Pembeli Impulsif PayLater & Retur Tinggi",
        "short_name": "Impulsif PayLater",
        "color": "#9B2C2C",
        "marketing_action": (
            "Sediakan panduan ukuran interaktif dan ulasan foto terverifikasi pada halaman "
            "fashion serta insentif tukar ukuran berbasis saldo toko (store credit) agar "
            "biaya logistik retur turun tanpa kehilangan volume PayLater."
        ),
    },
    5: {
        "name": "Member Baru Window-Shopper",
        "short_name": "Window-Shopper Baru",
        "color": "#805AD5",
        "marketing_action": (
            "Aktifkan alur onboarding terpandu dengan pengingat keranjang tertinggal "
            "(abandoned cart) otomatis disertai kupon transaksi pertama dan jaminan "
            "pengembalian mudah untuk mengonversi tingginya sesi browsing menjadi pembelian."
        ),
    },
}


def fit_final_gmm_and_order_segments(
    df_normal: pd.DataFrame,
    X_26d: np.ndarray,
    k: int = FINAL_K,
    seed: int = SEED,
) -> dict[str, Any]:
    """Latih GMM Full Covariance (26D, K=6) pada seluruh 118.200 pelanggan non-bot dan urutkan ID 0..5."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-2,
        n_init=5,
        max_iter=300,
        random_state=seed,
    ).fit(X_26d)

    raw_labels = gmm.predict(X_26d)
    proba = gmm.predict_proba(X_26d)
    log_lik = gmm.score_samples(X_26d)

    # Urutkan ID klaster mentah secara deterministik ke 0..5 sesuai ciri dominan segmen
    tmp = df_normal.copy()
    tmp["_raw"] = raw_labels
    raw_stats: dict[int, dict[str, float]] = {}
    for cid, grp in tmp.groupby("_raw"):
        raw_stats[int(cid)] = {
            "vch": float(grp["persen_voucher"].median()),
            "pyl": float(grp["persen_paylater"].median()),
            "rec": float(grp["recency_hari"].median()),
            "krj": float(grp["rata2_keranjang"].median()),
            "mbr": float(grp["lama_member_bulan"].median()),
            "pagi": float(grp["porsi_jam_pagi"].mean()),
        }

    remaining = set(raw_stats.keys())
    raw_to_canonical: dict[int, int] = {}

    def assign_best(target_id: int, metric: str, reverse: bool = True) -> None:
        chosen = sorted(remaining, key=lambda c: raw_stats[c][metric], reverse=reverse)[0]
        raw_to_canonical[chosen] = target_id
        remaining.remove(chosen)

    assign_best(4, "pyl", reverse=True)   # Impulsif PayLater & Retur Tinggi
    assign_best(3, "krj", reverse=True)   # Premium / Affluent iOS
    assign_best(2, "rec", reverse=True)   # Pasif / Dormant COD
    assign_best(5, "mbr", reverse=False)  # Member Baru Window-Shopper
    assign_best(0, "vch", reverse=True)   # Pemburu Promo & Flash Sale Malam
    assign_best(1, "pagi", reverse=True)  # Keluarga Pembeli Rutin Pagi

    labels = np.array([raw_to_canonical[int(c)] for c in raw_labels], dtype=np.int32)
    max_post = proba.max(axis=1)

    return {
        "gmm": gmm,
        "labels": labels,
        "max_posterior": max_post,
        "log_likelihood": log_lik,
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "mean_max_posterior": float(max_post.mean()),
        "frac_posterior_gt_0p9": float((max_post > 0.90).mean()),
    }


def build_segment_profiles(
    df_normal: pd.DataFrame,
    labels: np.ndarray,
    max_posterior: np.ndarray,
) -> list[dict[str, Any]]:
    """Bangun ringkasan profil perilaku, komposisi kategori/jam, dan aksi pemasaran tiap segmen."""
    tmp = df_normal.copy()
    tmp["_seg"] = labels
    tmp["_post"] = max_posterior
    overall_kat = df_normal[list(PORSI_KATEGORI_COLUMNS)].mean()
    total_spend = float(df_normal["belanja_90h"].sum())

    profiles: list[dict[str, Any]] = []
    for seg_id in range(FINAL_K):
        grp = tmp[tmp["_seg"] == seg_id]
        meta = SEGMENT_CATALOG[seg_id]
        kat_mean = grp[list(PORSI_KATEGORI_COLUMNS)].mean()
        lift = (kat_mean / overall_kat).sort_values(ascending=False)
        top_cats = [
            {
                "category": str(c.replace("porsi_", "")),
                "mean_share": round(float(kat_mean[c]), 4),
                "lift_vs_avg": round(float(lift[c]), 2),
            }
            for c in lift.index[:3]
        ]
        jam_mean = grp[list(PORSI_JAM_COLUMNS)].mean()
        peak_jam = str(jam_mean.idxmax()).replace("porsi_jam_", "")

        profiles.append(
            {
                "cluster": int(seg_id),
                "name": meta["name"],
                "short_name": meta["short_name"],
                "n_customers": int(len(grp)),
                "pct_of_normal": round(float(len(grp) / len(df_normal) * 100.0), 2),
                "pct_of_total_120k": round(float(len(grp) / 120_000.0 * 100.0), 2),
                "revenue_share_pct": round(float(grp["belanja_90h"].sum() / total_spend * 100.0), 2),
                "mean_posterior": round(float(grp["_post"].mean()), 4),
                "f0_pct": round(float((grp["frekuensi_90h"] == 0).mean() * 100.0), 2),
                "median_recency_hari": round(float(grp["recency_hari"].median()), 1),
                "median_frekuensi_90h": round(float(grp["frekuensi_90h"].median()), 1),
                "median_belanja_90h_idr": round(float(grp["belanja_90h"].median()), 0),
                "median_keranjang_idr": round(float(grp["rata2_keranjang"].median()), 0),
                "median_item_per_pesanan": round(float(grp["item_per_pesanan"].median()), 1),
                "median_persen_voucher": round(float(grp["persen_voucher"].median()), 3),
                "median_persen_flash_sale": round(float(grp["persen_flash_sale"].median()), 3),
                "median_persen_paylater": round(float(grp["persen_paylater"].median()), 3),
                "median_persen_cod": round(float(grp["persen_cod"].median()), 3),
                "mean_retur_90h": round(float(grp["retur_90h"].mean()), 2),
                "mean_ulasan_90h": round(float(grp["ulasan_90h"].mean()), 2),
                "median_sesi_per_minggu": round(float(grp["sesi_per_minggu"].median()), 2),
                "median_rasio_konversi": round(float(grp["rasio_konversi"].median()), 3),
                "median_lama_member_bulan": round(float(grp["lama_member_bulan"].median()), 1),
                "median_umur": round(float(grp["umur"].median()), 1),
                "ios_pct": round(float((grp["perangkat"] == "ios").mean() * 100.0), 1),
                "peak_time_slot": peak_jam,
                "peak_time_share": round(float(jam_mean.max()), 3),
                "time_shares": {
                    c.replace("porsi_jam_", ""): round(float(jam_mean[c]), 3)
                    for c in PORSI_JAM_COLUMNS
                },
                "top_lift_categories": top_cats,
                "marketing_action": meta["marketing_action"],
            }
        )
    return profiles


def buat_grafik_profil_segmen(
    df_normal: pd.DataFrame,
    df_canon: pd.DataFrame,
    Z_pca: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """Buat grafik 2-panel Tugas 3f: heatmap profil z-score segmen dan proyeksi PCA 2D."""
    configure_plotting()
    fig, axes = plt.subplots(
        1, 2, figsize=(LEBAR_GAMBAR, 3.35), gridspec_kw={"width_ratios": [1.32, 1.0]}, constrained_layout=True
    )

    # Panel A: Heatmap profil z-score 6 segmen pada 12 fitur kunci
    ax = axes[0]
    ax.grid(False)
    feat_specs = [
        ("log_recency_hari", "Recency"),
        ("log1p_frekuensi_90h", "Frekuensi"),
        ("log_rata2_keranjang", "Keranjang"),
        ("log1p_item_per_pesanan", "Item/pesan"),
        ("logit_persen_voucher", "Voucher"),
        ("logit_persen_flash_sale", "Flash sale"),
        ("logit_persen_paylater", "PayLater"),
        ("logit_persen_cod", "COD"),
        ("log1p_retur_90h", "Retur"),
        ("log_sesi_per_minggu", "Sesi/mg"),
        ("log_lama_member_bulan", "Lama mbr"),
        ("is_ios", "iOS"),
    ]
    z_all = StandardScaler().fit_transform(
        df_canon[[c for c, _ in feat_specs]].to_numpy(dtype=np.float64)
    )
    hm = np.zeros((FINAL_K, len(feat_specs)), dtype=np.float64)
    for cid in range(FINAL_K):
        hm[cid] = z_all[labels == cid].mean(axis=0)

    im = ax.imshow(hm, cmap="RdBu_r", vmin=-1.3, vmax=1.3, aspect="auto")
    ax.set_xticks(np.arange(len(feat_specs)))
    ax.set_xticklabels([lbl for _, lbl in feat_specs], rotation=38, ha="right", fontsize=5.8)
    ax.set_yticks(np.arange(FINAL_K))
    ax.set_yticklabels(
        [f"C{i}: {SEGMENT_CATALOG[i]['short_name']}" for i in range(FINAL_K)], fontsize=6.0
    )
    for i in range(FINAL_K):
        for j in range(len(feat_specs)):
            val = float(hm[i, j])
            val_disp = 0.0 if abs(val) < 0.05 else val
            col_txt = "white" if abs(val) >= 0.78 else "#1A202C"
            ax.text(j, i, angka_id(val_disp, 1), ha="center", va="center", fontsize=5.1, color=col_txt)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.ax.tick_params(labelsize=5.8)
    cbar.set_label("Rata-rata z-score", fontsize=6.0)
    sumbu_koma(cbar.ax, "y")
    ax.set_title("(a) Profil Standar (z-score) 6 Segmen Pelanggan")

    # Panel B: Proyeksi PCA-2D (PC1 vs PC2) subsampel 12.000 pelanggan + sentroid berlabel
    ax = axes[1]
    rng = np.random.default_rng(SEED)
    idx_s = rng.choice(len(Z_pca), size=12_000, replace=False)
    for cid in range(FINAL_K):
        mask_s = labels[idx_s] == cid
        meta = SEGMENT_CATALOG[cid]
        ax.scatter(
            Z_pca[idx_s[mask_s], 0],
            Z_pca[idx_s[mask_s], 1],
            s=3.0,
            alpha=0.24,
            color=meta["color"],
            rasterized=True,
        )
    # Sentroid berlabel dengan offset khusus agar C0 & C4 (terpisah di PC4) tidak bertumpuk
    label_offsets: dict[int, tuple[float, float]] = {
        0: (16.0, -14.0),
        1: (-10.0, 10.0),
        2: (0.0, -16.0),
        3: (-12.0, -14.0),
        4: (-2.0, 14.0),
        5: (-14.0, -12.0),
    }
    for cid in range(FINAL_K):
        mask_all = labels == cid
        cx, cy = float(Z_pca[mask_all, 0].mean()), float(Z_pca[mask_all, 1].mean())
        meta = SEGMENT_CATALOG[cid]
        ax.scatter([cx], [cy], s=42, color=meta["color"], edgecolors="white", linewidths=0.9, zorder=5)
        ox, oy = label_offsets[cid]
        ax.annotate(
            f"C{cid}",
            (cx, cy),
            xytext=(ox, oy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color="#1A202C",
            arrowprops={"arrowstyle": "-", "color": meta["color"], "lw": 0.8},
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": meta["color"], "alpha": 0.95, "lw": 0.8},
            zorder=6,
        )
    ax.set_xlabel(f"{miring('Principal Component')} 1 (belanja vs promo)")
    ax.set_ylabel(f"{miring('Principal Component')} 2 (aktif vs pasif)")
    ax.set_title("(b) Proyeksi PCA 2D & Sentroid 6 Segmen")
    sumbu_koma(ax, "xy")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
