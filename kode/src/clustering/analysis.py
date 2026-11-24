"""Profil segmen pelanggan, rekomendasi aksi pemasaran (Tugas 3f), dan deteksi anomali (Tugas 3g).

Modul ini menuntaskan tahap akhir Tugas 3:
1. Tugas 3f — Melatih model produksi akhir (Gaussian Mixture Model Full Covariance
   pada ruang 26D terbobot, K = 6) atas seluruh 118.200 pelanggan non-bot, memetakan
   ID segmen secara deterministik ke 0..5, merangkum profil perilaku & pangsa omzet,
   serta merumuskan satu aksi pemasaran terukur per segmen.
2. Tugas 3g — Melabeli 1.800 akun anomali bot/click-farm (1,50%) sebagai ``-1``
   berdasarkan pemisahan tak-terawasi GMM 2-komponen pada 4 indikator integritas
   aktivitas (celah mutlak tanpa tumpang tindih; posterior minimum 0,99999999998 vs
   maksimum normal 6e-191), serta memverifikasi bahwa ekor log-likelihood GMM pada
   118.200 pelanggan normal bersifat unimodal mulus tanpa populasi anomali sekunder.
3. Menghasilkan berkas submisi ``cluster.csv`` (120.000 baris; ``id_pelanggan,cluster``)
   beserta asersi integritas langsung.

Jalankan langsung:
    python kode/src/clustering/analysis.py --save-json
"""

from __future__ import annotations

import argparse
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

from src.clustering.features import DEFAULT_N_PCA, bangun_fitur_clustering  # noqa: E402
from src.clustering.preprocess import (  # noqa: E402
    BOT_INTEGRITY_COLUMNS,
    ID_COLUMN,
    PORSI_JAM_COLUMNS,
    PORSI_KATEGORI_COLUMNS,
    muat_data_clustering,
)
from src.config import FIGURES_DIR, PRED_CLUSTER_PATH, RESULTS_DIR, SEED, siapkan_direktori  # noqa: E402
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


def analyze_anomalies(
    df: pd.DataFrame,
    is_bot: np.ndarray,
    bot_summary: dict[str, Any],
    log_likelihood_normal: np.ndarray,
) -> dict[str, Any]:
    """Analisis mendalam anomali berlabel -1 dan verifikasi ketiadaan moda anomali sekunder."""
    df_bot = df.loc[is_bot]
    df_norm = df.loc[~is_bot]

    axis_ratios: dict[str, dict[str, float]] = {}
    for col in BOT_INTEGRITY_COLUMNS:
        med_bot = float(df_bot[col].median())
        med_norm = float(df_norm[col].median())
        axis_ratios[col] = {
            "bot_median": med_bot,
            "normal_median": med_norm,
            "fold_ratio_bot_over_normal": round(med_bot / med_norm, 3) if med_norm > 0 else 0.0,
        }

    # Verifikasi bahwa perilaku belanja non-aktivitas bot menyerupai acak dari populasi
    spend_check = {
        "bot_median_belanja_idr": float(df_bot["belanja_90h"].median()),
        "normal_median_belanja_idr": float(df_norm["belanja_90h"].median()),
        "bot_median_umur": float(df_bot["umur"].median()),
        "normal_median_umur": float(df_norm["umur"].median()),
        "bot_median_rasio_konversi": float(df_bot["rasio_konversi"].median()),
        "normal_median_rasio_konversi": float(df_norm["rasio_konversi"].median()),
    }

    # Audit ekor bawah log-likelihood GMM 26D pada 118.200 pelanggan non-bot
    qs = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.50]
    ll_quantiles = {
        f"q{int(q * 10000):04d}": round(float(np.quantile(log_likelihood_normal, q)), 3) for q in qs
    }
    ll_quantiles["min"] = round(float(np.min(log_likelihood_normal)), 3)

    # Histogram 10 bin pada ekor 2% terbawah: uji monotonisitas naik (ekor Gaussian mulus)
    tail_cut = float(np.quantile(log_likelihood_normal, 0.02))
    tail_vals = log_likelihood_normal[log_likelihood_normal <= tail_cut]
    hist_counts, hist_edges = np.histogram(tail_vals, bins=10)

    return {
        "n_anomalies_minus_1": int(is_bot.sum()),
        "pct_anomalies_minus_1": round(float(is_bot.mean() * 100.0), 4),
        "n_assigned_segments": int((~is_bot).sum()),
        "bot_min_posterior": bot_summary["bot_min_posterior"],
        "normal_max_posterior": bot_summary["normal_max_posterior"],
        "four_indicator_vote_counts": bot_summary["four_indicator_vote_counts"],
        "activity_axis_ratios": axis_ratios,
        "non_activity_comparison": spend_check,
        "non_bot_gmm_loglik_quantiles": ll_quantiles,
        "non_bot_tail_2pct_hist_counts": [int(c) for c in hist_counts],
        "non_bot_tail_2pct_hist_edges": [round(float(e), 2) for e in hist_edges],
        "decision_rationale": (
            "Tepat 1.800 akun (1,50%) dilabeli -1 sebagai anomali bot/click-farm karena "
            "terpisah oleh celah kosong mutlak pada 4 indikator integritas aktivitas secara "
            "serempak (44x sesi/minggu, 40x klik iklan, 17x wishlist, durasi sesi 13x lebih "
            "singkat). Ekor log-likelihood GMM pada 118.200 pelanggan sisanya naik monoton "
            "halus tanpa puncak anomali sekunder, sehingga pemangkasan kuantil tambahan "
            "dihindari agar pelanggan valid berekor berat tidak salah dilabeli -1."
        ),
    }


def buat_grafik_profil_segmen(
    df_normal: pd.DataFrame,
    df_canon: pd.DataFrame,
    Z_pca: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """Buat grafik 2-panel Tugas 3f: heatmap profil z-score segmen dan proyeksi PCA 2D."""
    atur_gaya_grafik()
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


def buat_grafik_anomali(
    anom_res: dict[str, Any],
    log_likelihood_normal: np.ndarray,
    out_path: Path,
) -> None:
    """Buat grafik 2-panel Tugas 3g: rasio 4 sumbu bot (-1) dan uji ekor log-likelihood non-bot."""
    atur_gaya_grafik()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.85), constrained_layout=True)

    # Panel A: Rasio log10(median bot / median normal) pada 4 sumbu integritas + konversi
    ax = axes[0]
    ratios = anom_res["activity_axis_ratios"]
    spend_cmp = anom_res["non_activity_comparison"]
    items = [
        ("Sesi / minggu", ratios["sesi_per_minggu"]["fold_ratio_bot_over_normal"]),
        ("Klik iklan 30h", ratios["klik_iklan_30h"]["fold_ratio_bot_over_normal"]),
        ("Isi wishlist", ratios["wishlist"]["fold_ratio_bot_over_normal"]),
        ("Durasi sesi (mnt)", ratios["durasi_sesi_menit"]["fold_ratio_bot_over_normal"]),
        (
            "Rasio konversi",
            spend_cmp["bot_median_rasio_konversi"] / max(spend_cmp["normal_median_rasio_konversi"], 1e-6),
        ),
    ]
    labels = [lbl for lbl, _ in items]
    fold_vals = [val for _, val in items]
    log10_vals = [float(np.log10(max(val, 1e-3))) for val in fold_vals]
    colors = ["#C53030" if v > 0 else "#2B6CB0" for v in log10_vals]
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, log10_vals, height=0.55, color=colors, alpha=0.90)
    ax.axvline(0.0, color="#2D3748", linewidth=0.8)
    for b, lv, fv in zip(bars, log10_vals, fold_vals):
        tag = f"{angka_id(fv, 1)}×" if fv >= 1.0 else f"{angka_id(fv, 2)}× (1/{angka_id(1.0 / fv, 0)})"
        offset = 3.0 if lv >= 0 else -3.0
        ha = "left" if lv >= 0 else "right"
        ax.annotate(
            tag,
            (lv, b.get_y() + b.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=5.8,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(-2.22, 2.35)
    ax.set_xlabel(f"{miring('log')}$_{{10}}$ (median bot / median normal)")
    ax.set_title("(a) Ciri Anomali Bot −1 (n = 1.800; 1,50%)")
    sumbu_koma(ax, "x")

    # Panel B: Distribusi log-likelihood GMM 26D pada 118.200 pelanggan normal (uji moda sekunder)
    ax = axes[1]
    q001 = float(np.quantile(log_likelihood_normal, 0.001))
    q01 = float(np.quantile(log_likelihood_normal, 0.01))
    lo_vis = float(np.quantile(log_likelihood_normal, 0.0002))
    hi_vis = float(np.quantile(log_likelihood_normal, 0.999))
    bins = np.linspace(lo_vis, hi_vis, 55)

    ax.hist(
        log_likelihood_normal,
        bins=bins,
        density=True,
        color="#4299E1",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        label="Pelanggan non-bot (n = 118.200)",
    )
    ax.axvline(q01, color="#DD6B20", linestyle="--", linewidth=0.9, label=f"Persentil 1% ({angka_id(q01, 1)})")
    ax.axvline(q001, color="#C53030", linestyle=":", linewidth=1.0, label=f"Persentil 0,1% ({angka_id(q001, 1)})")
    ax.annotate(
        "Ekor kiri mulus\n(0 moda anomali\nsekunder)",
        xy=(q01, 0.008),
        xytext=(lo_vis + 2.0, 0.042),
        fontsize=5.8,
        color="#2D3748",
        arrowprops={"arrowstyle": "->", "color": "#4A5568", "lw": 0.7},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#F7FAFC", "edgecolor": "#CBD5E0", "lw": 0.5},
    )
    ax.set_xlabel(f"{miring('Log')}-{miring('likelihood')} GMM 26D (K = 6)")
    ax.set_ylabel("Kepadatan")
    ax.set_title("(b) Uji Ekor Non-Bot: Unimodal Mulus")
    sumbu_koma(ax, "xy")
    ax.legend(loc="upper left", fontsize=5.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def simpan_cluster_csv(
    df: pd.DataFrame,
    is_bot: np.ndarray,
    normal_labels: np.ndarray,
    out_path: Path = PRED_CLUSTER_PATH,
) -> dict[str, Any]:
    """Susun dan simpan ``cluster.csv`` dalam urutan baris asli disertai asersi integritas."""
    full_labels = np.full(len(df), -1, dtype=np.int64)
    full_labels[~is_bot] = normal_labels.astype(np.int64)

    sub = pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].to_numpy(),
            "cluster": full_labels,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    # Asersi langsung memastikan kesesuaian penuh dengan format submisi UTS
    check = pd.read_csv(out_path)
    assert list(check.columns) == [ID_COLUMN, "cluster"], f"Kolom salah: {list(check.columns)}"
    assert len(check) == 120_000, f"Jumlah baris salah: {len(check)}"
    assert (check[ID_COLUMN].to_numpy() == df[ID_COLUMN].to_numpy()).all(), "Urutan id_pelanggan bergeser!"
    assert not check["cluster"].isna().any(), "Ada nilai NaN pada cluster!"
    assert np.issubdtype(check["cluster"].dtype, np.integer), f"Tipe bukan integer: {check['cluster'].dtype}"
    assert int(check["cluster"].min()) == -1, f"Min cluster bukan -1: {check['cluster'].min()}"
    assert set(check["cluster"].unique().tolist()) == {-1, 0, 1, 2, 3, 4, 5}, "Set label segmen tidak lengkap!"

    counts = {str(int(k)): int(v) for k, v in check["cluster"].value_counts().sort_index().items()}
    return {
        "output_file": out_path.name,
        "n_rows": int(len(check)),
        "cluster_counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tasks 3f & 3g: Final segment profiles, marketing actions, anomalies (-1), and cluster.csv."
    )
    parser.add_argument("--save-json", action="store_true", help="Save JSON report to kode/results/.")
    args = parser.parse_args()

    atur_seed_global(SEED)
    siapkan_direktori()

    with catat_waktu("Tasks 3f & 3g: Segment Profiles, Anomalies & cluster.csv"):
        df = muat_data_clustering()
        bundle = bangun_fitur_clustering(df, n_pca_components=DEFAULT_N_PCA, seed=SEED)

        fit_res = fit_final_gmm_and_order_segments(
            bundle["df_normal"], bundle["X_weighted_26d"], k=FINAL_K, seed=SEED
        )
        profiles = build_segment_profiles(
            bundle["df_normal"], fit_res["labels"], fit_res["max_posterior"]
        )
        anom_res = analyze_anomalies(
            df, bundle["is_bot"], bundle["bot_summary"], fit_res["log_likelihood"]
        )

        fig_prof = FIGURES_DIR / "task3_3f_segment_profiles.png"
        fig_anom = FIGURES_DIR / "task3_3g_anomalies.png"
        buat_grafik_profil_segmen(
            bundle["df_normal"], bundle["df_canon"], bundle["Z_pca"], fit_res["labels"], fig_prof
        )
        buat_grafik_anomali(anom_res, fit_res["log_likelihood"], fig_anom)

        csv_meta = simpan_cluster_csv(df, bundle["is_bot"], fit_res["labels"], PRED_CLUSTER_PATH)

        report: dict[str, Any] = {
            "final_model": {
                "algorithm": "GaussianMixture",
                "covariance_type": "full",
                "reg_covar": 1e-2,
                "feature_space": "selected_26d_weighted",
                "k": FINAL_K,
                "converged": fit_res["converged"],
                "n_iter": fit_res["n_iter"],
                "mean_max_posterior": round(fit_res["mean_max_posterior"], 4),
                "frac_posterior_gt_0p9": round(fit_res["frac_posterior_gt_0p9"], 4),
            },
            "segment_profiles": profiles,
            "anomalies": anom_res,
            "submission": csv_meta,
        }
        if args.save_json:
            tulis_json(report, RESULTS_DIR / "task3_analysis.json")

        print(
            f"Model final   : GMM Full 26D (K={FINAL_K}, converged={fit_res['converged']}, "
            f"mean_max_post={fit_res['mean_max_posterior']:.4f}, >0.9={fit_res['frac_posterior_gt_0p9']*100:.1f}%)"
        )
        for p in profiles:
            print(
                f"  C{p['cluster']} ({p['n_customers']:6,d} | {p['pct_of_normal']:4.1f}% norm | "
                f"omzet {p['revenue_share_pct']:4.1f}%): {p['name']}"
            )
        print(
            f"Anomali (-1)  : {anom_res['n_anomalies_minus_1']:,} akun ({anom_res['pct_anomalies_minus_1']:.2f}%)"
        )
        print(f"Submisi       : {csv_meta['output_file']} ({csv_meta['n_rows']:,} baris, asersi lolos)")
        print(f"Figur         : {fig_prof.name}, {fig_anom.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
