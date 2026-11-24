"""Konstanta skema, audit kualitas data, dan prapemrosesan Tugas 3 (Segmentasi Pelanggan).

Modul ini menangani seluruh aspek prapemrosesan (Tugas 3a) pada ``clustering.csv.gz``
(120.000 pelanggan, 60 kolom):
1. Deteksi tak-terawasi pra-skala untuk anomali bot/click-farm (1.800 akun; 1,50%)
   menggunakan GMM 2-komponen pada ruang integritas aktivitas
   (``sesi_per_minggu``, ``durasi_sesi_menit``, ``klik_iklan_30h``, ``wishlist``),
   sehingga ekstrem bot tidak mendistorsi parameter skala, imputasi, maupun sentroid.
2. Transformasi kanonik yang selaras dengan keluarga distribusi masing-masing variabel
   untuk menormalkan skala & kemencengan berat tanpa menciptakan jurang nol (*zero-cliff*):
   - Variabel positif menceng kanan (``recency_hari``, ``rata2_keranjang``,
     ``sesi_per_minggu``, ``lama_member_bulan``): ``log`` setelah winsorisasi halus
     [0,1%, 99,9%].
   - Variabel cacah (*count*: ``frekuensi_90h``, ``item_per_pesanan``, ``retur_90h``,
     ``ulasan_90h``): ``log1p``.
   - Variabel proporsi pada [0, 1] (``persen_voucher``, ``persen_flash_sale``,
     ``persen_paylater``, ``persen_cod``): ``logit(clip(p, 0,01, 0,99))``.
   - Variabel simetris (``umur``): standardisasi linear.
3. Imputasi nilai hilang struktural pada ``rata2_keranjang`` (6.736 baris = 100% cocok
   dengan ``frekuensi_90h == 0``) memakai regresi Ridge pada ruang ``log(rata2_keranjang)``
   berbasis perilaku non-frekuensi & komposisi belanja, mencegah jurang 11 satuan log
   yang muncul bila nilai kosong diisi 0.
4. Transformasi data komposisional (*Centered Log-Ratio* / CLR) pada 16 kolom
   ``porsi_<kategori>`` dan 4 kolom ``porsi_jam_*`` (membuang 1 derajat kebebasan
   kolinear pada blok jam) untuk memetakan simpleks Aitchison ke ruang Euklides.
5. Pengkodean kategorikal berbasis sinyal: ``perangkat`` dikodekan menjadi indikator
   tunggal ``is_ios`` karena perilaku ``web`` identik dengan ``android``, sedangkan
   ``kota_tier`` terbukti seragam tanpa sinyal lintas tier.

Jalankan langsung:
    python kode/src/clustering/preprocess.py --save-json
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
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import CLUSTERING_PATH, FIGURES_DIR, RESULTS_DIR, SEED, siapkan_direktori  # noqa: E402
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

ID_COLUMN = "id_pelanggan"

# 16 kolom komposisional kategori belanja (jumlah baris = 1,0)
PORSI_KATEGORI_COLUMNS: tuple[str, ...] = (
    "porsi_fashion",
    "porsi_elektronik",
    "porsi_kecantikan",
    "porsi_rumah_tangga",
    "porsi_makanan_minuman",
    "porsi_ibu_bayi",
    "porsi_olahraga",
    "porsi_hobi",
    "porsi_otomotif",
    "porsi_kesehatan",
    "porsi_buku_alat_tulis",
    "porsi_voucher_digital",
    "porsi_gadget_aksesoris",
    "porsi_perabot",
    "porsi_pertanian",
    "porsi_lainnya",
)

# 9 kategori dengan sinyal segmen nyata (Spearman sum_r2 >= 0,045)
INFORMATIVE_KATEGORI_COLUMNS: tuple[str, ...] = (
    "porsi_fashion",
    "porsi_elektronik",
    "porsi_kecantikan",
    "porsi_rumah_tangga",
    "porsi_makanan_minuman",
    "porsi_ibu_bayi",
    "porsi_voucher_digital",
    "porsi_gadget_aksesoris",
    "porsi_perabot",
)

LOW_SIGNAL_KATEGORI_COLUMNS: tuple[str, ...] = tuple(
    c for c in PORSI_KATEGORI_COLUMNS if c not in INFORMATIVE_KATEGORI_COLUMNS
)

# 4 kolom komposisional waktu belanja (jumlah baris = 1,0)
PORSI_JAM_COLUMNS: tuple[str, ...] = (
    "porsi_jam_pagi",
    "porsi_jam_siang",
    "porsi_jam_sore",
    "porsi_jam_malam",
)

# 15 kolom CRM anonim (kamus_data.md hanya mencantumkan n01-n10; berkas memuat n01-n15)
ANONYMOUS_N_COLUMNS: tuple[str, ...] = tuple(f"n{i:02d}" for i in range(1, 16))

# 4 kolom integritas aktivitas yang memisahkan akun bot/click-farm secara mutlak
BOT_INTEGRITY_COLUMNS: tuple[str, ...] = (
    "sesi_per_minggu",
    "durasi_sesi_menit",
    "klik_iklan_30h",
    "wishlist",
)

# Kelompok transformasi kanonik fitur perilaku inti (non-noise, non-redundan)
LOG_POSITIVE_COLUMNS: tuple[str, ...] = (
    "recency_hari",
    "rata2_keranjang",
    "sesi_per_minggu",
    "lama_member_bulan",
)
LOG1P_COUNT_COLUMNS: tuple[str, ...] = (
    "frekuensi_90h",
    "item_per_pesanan",
    "retur_90h",
    "ulasan_90h",
)
LOGIT_PROPORTION_COLUMNS: tuple[str, ...] = (
    "persen_voucher",
    "persen_flash_sale",
    "persen_paylater",
    "persen_cod",
)
LINEAR_NUMERIC_COLUMNS: tuple[str, ...] = ("umur",)

# Kolom bernama tanpa sinyal segmen pada pelanggan non-bot
NOISE_NAMED_COLUMNS: tuple[str, ...] = (
    "kota_tier",
    "jumlah_alamat",
    "rating_beri_rata2",
    "durasi_sesi_menit",
    "klik_iklan_30h",
    "wishlist",
)

# Kolom deterministik / redundan (kolinear tinggi dengan fitur inti)
REDUNDANT_COLUMNS: tuple[str, ...] = (
    "belanja_90h",
    "belanja_30h",
    "frekuensi_30h",
    "rasio_konversi",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = ("kota_tier", "perangkat")

# Konstanta pseudocount untuk transformasi logit dan CLR
LOGIT_EPS = 0.01
CLR_EPS = 1e-3
WINSOR_LO = 0.001
WINSOR_HI = 0.999


def muat_data_clustering(path: Path = CLUSTERING_PATH) -> pd.DataFrame:
    """Muat berkas ``clustering.csv.gz`` dalam urutan baris asli."""
    return pd.read_csv(path)


def logit_halus(values: np.ndarray, eps: float = LOGIT_EPS) -> np.ndarray:
    """Transformasi logit terklip pada [eps, 1 - eps] untuk variabel proporsi Beta."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def transformasi_clr(shares: np.ndarray, eps: float = CLR_EPS) -> np.ndarray:
    """Centered Log-Ratio (CLR) untuk data komposisional pada simpleks unit."""
    arr = np.maximum(np.asarray(shares, dtype=np.float64), 0.0) + eps
    log_arr = np.log(arr)
    return log_arr - np.mean(log_arr, axis=1, keepdims=True)


def winsorisasi_kuantil(
    values: np.ndarray,
    lo_q: float = WINSOR_LO,
    hi_q: float = WINSOR_HI,
) -> tuple[np.ndarray, float, float]:
    """Klip ekor ekstrem pada rentang kuantil [lo_q, hi_q] dengan mengabaikan NaN."""
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.nanquantile(arr, lo_q))
    hi = float(np.nanquantile(arr, hi_q))
    return np.clip(arr, lo, hi), lo, hi


def deteksi_anomali_bot_aktivitas(
    df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pisahkan akun bot/click-farm secara tak-terawasi pada ruang integritas aktivitas.

    Memakai Gaussian Mixture 2-komponen pada ``log1p`` dari empat fitur aktivitas
    (``sesi_per_minggu``, ``durasi_sesi_menit``, ``klik_iklan_30h``, ``wishlist``)
    yang telah distandardisasi. Komponen minoritas otomatis mengisolasi akun bot
    tanpa ambang keras manual.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Masker boolean ``is_bot`` (panjang N) dan ringkasan statistik pemisahan.
    """
    act = np.column_stack(
        [np.log1p(np.maximum(df[c].to_numpy(dtype=np.float64), 0.0)) for c in BOT_INTEGRITY_COLUMNS]
    )
    act_scaled = StandardScaler().fit_transform(act)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=5,
        random_state=seed,
    ).fit(act_scaled)
    labels = gmm.predict(act_scaled)
    proba = gmm.predict_proba(act_scaled)
    counts = np.bincount(labels)
    minor_comp = int(np.argmin(counts))
    is_bot = labels == minor_comp
    bot_post = proba[is_bot, minor_comp]
    normal_post = proba[~is_bot, minor_comp]

    # Catat celah kosong (*empty gap*) pada keempat variabel integritas
    gaps: dict[str, dict[str, float]] = {}
    for col in BOT_INTEGRITY_COLUMNS:
        s_bot = df.loc[is_bot, col]
        s_norm = df.loc[~is_bot, col]
        gaps[col] = {
            "bot_min": float(s_bot.min()),
            "bot_median": float(s_bot.median()),
            "bot_max": float(s_bot.max()),
            "normal_min": float(s_norm.min()),
            "normal_q001": float(s_norm.quantile(0.001)),
            "normal_median": float(s_norm.median()),
            "normal_q999": float(s_norm.quantile(0.999)),
            "normal_max": float(s_norm.max()),
        }

    # Verifikasi konsistensi suara 4 indikator ekstrem
    v1 = df["sesi_per_minggu"] > 50.0
    v2 = df["durasi_sesi_menit"] < 1.2
    v3 = df["klik_iklan_30h"] > 60.0
    v4 = df["wishlist"] > 95.0
    votes = (v1.astype(int) + v2.astype(int) + v3.astype(int) + v4.astype(int)).value_counts().sort_index()

    summary: dict[str, Any] = {
        "n_total": int(len(df)),
        "n_bot": int(is_bot.sum()),
        "pct_bot": float(is_bot.mean() * 100.0),
        "n_normal": int((~is_bot).sum()),
        "bot_min_posterior": float(bot_post.min()),
        "normal_max_posterior": float(normal_post.max()),
        "four_indicator_vote_counts": {str(int(k)): int(v) for k, v in votes.items()},
        "gaps": gaps,
    }
    return is_bot, summary


def praproses_fitur_kanonik(
    df_normal: pd.DataFrame,
    ridge_alpha: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Terapkan transformasi kanonik (log, log1p, logit, CLR, imputasi Ridge) pada pelanggan non-bot.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        DataFrame fitur tertransformasi (belum distandardisasi, indeks mengikuti ``df_normal``)
        beserta ringkasan diagnostik transformasi, kemencengan, dan imputasi.
    """
    out = pd.DataFrame(index=df_normal.index)
    skew_before: dict[str, float] = {}
    skew_after: dict[str, float] = {}
    winsor_bounds: dict[str, list[float]] = {}

    # 1. Positif menceng kanan -> winsorize [0,1%, 99,9%] lalu log
    for col in LOG_POSITIVE_COLUMNS:
        raw = df_normal[col].to_numpy(dtype=np.float64)
        skew_before[col] = float(stats.skew(raw[~np.isnan(raw)]))
        clipped, lo, hi = winsorisasi_kuantil(raw)
        winsor_bounds[col] = [lo, hi]
        if col == "rata2_keranjang":
            # Tunda imputasi hingga fitur prediktor lain tersedia
            out["log_rata2_keranjang"] = np.where(np.isnan(clipped), np.nan, np.log(clipped))
        else:
            transformed = np.log(clipped)
            out[f"log_{col}"] = transformed
            skew_after[col] = float(stats.skew(transformed))

    # 2. Cacah (count) -> winsorize ujung atas 99,9% lalu log1p
    for col in LOG1P_COUNT_COLUMNS:
        raw = df_normal[col].to_numpy(dtype=np.float64)
        skew_before[col] = float(stats.skew(raw))
        hi = float(np.quantile(raw, WINSOR_HI))
        winsor_bounds[col] = [0.0, hi]
        transformed = np.log1p(np.clip(raw, 0.0, hi))
        out[f"log1p_{col}"] = transformed
        skew_after[col] = float(stats.skew(transformed))

    # 3. Proporsi [0, 1] -> logit halus
    for col in LOGIT_PROPORTION_COLUMNS:
        raw = df_normal[col].to_numpy(dtype=np.float64)
        skew_before[col] = float(stats.skew(raw))
        transformed = logit_halus(raw)
        out[f"logit_{col}"] = transformed
        skew_after[col] = float(stats.skew(transformed))

    # 4. Numerik simetris
    for col in LINEAR_NUMERIC_COLUMNS:
        raw = df_normal[col].to_numpy(dtype=np.float64)
        skew_before[col] = float(stats.skew(raw))
        out[col] = raw
        skew_after[col] = float(stats.skew(raw))

    # 5. Kategorikal perangkat -> indikator tunggal is_ios (web == android secara perilaku)
    out["is_ios"] = (df_normal["perangkat"] == "ios").astype(np.float64)

    # 6. Komposisional waktu belanja (CLR 4 -> ambil 3 dimensi independen: pagi, siang, malam)
    clr_jam = transformasi_clr(df_normal[list(PORSI_JAM_COLUMNS)].to_numpy(dtype=np.float64))
    out["clr_jam_pagi"] = clr_jam[:, 0]
    out["clr_jam_siang"] = clr_jam[:, 1]
    out["clr_jam_malam"] = clr_jam[:, 3]

    # 7. Komposisional 16 kategori belanja (CLR)
    clr_kat = transformasi_clr(df_normal[list(PORSI_KATEGORI_COLUMNS)].to_numpy(dtype=np.float64))
    for i, col in enumerate(PORSI_KATEGORI_COLUMNS):
        out[f"clr_{col}"] = clr_kat[:, i]

    # 8. Imputasi struktural log(rata2_keranjang) memakai Ridge pada perilaku non-frekuensi
    pred_cols = [
        "log_recency_hari",
        "log1p_item_per_pesanan",
        "log1p_retur_90h",
        "log_sesi_per_minggu",
        "log_lama_member_bulan",
        "umur",
        "logit_persen_voucher",
        "logit_persen_flash_sale",
        "logit_persen_paylater",
        "logit_persen_cod",
        "is_ios",
        "clr_jam_pagi",
        "clr_jam_siang",
        "clr_jam_malam",
    ] + [f"clr_{c}" for c in PORSI_KATEGORI_COLUMNS]

    obs_mask = out["log_rata2_keranjang"].notna().to_numpy()
    X_pred = StandardScaler().fit_transform(out[pred_cols].to_numpy(dtype=np.float64))
    y_obs = out.loc[obs_mask, "log_rata2_keranjang"].to_numpy(dtype=np.float64)

    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_pred[obs_mask], y_obs)
    r2_obs = float(ridge.score(X_pred[obs_mask], y_obs))
    pred_missing = ridge.predict(X_pred[~obs_mask])
    # Pastikan nilai imputasi tetap berada dalam rentang teramati
    lo_log, hi_log = float(y_obs.min()), float(y_obs.max())
    pred_missing = np.clip(pred_missing, lo_log, hi_log)

    krj_imputed = out["log_rata2_keranjang"].to_numpy(dtype=np.float64).copy()
    krj_imputed[~obs_mask] = pred_missing
    out["log_rata2_keranjang"] = krj_imputed
    skew_after["rata2_keranjang"] = float(stats.skew(krj_imputed))

    meta: dict[str, Any] = {
        "n_rows": int(len(df_normal)),
        "n_missing_rata2_keranjang": int((~obs_mask).sum()),
        "pct_missing_rata2_keranjang": float((~obs_mask).mean() * 100.0),
        "ridge_imputation_r2": r2_obs,
        "imputed_median_idr": float(np.exp(np.median(pred_missing))),
        "observed_median_idr": float(np.exp(np.median(y_obs))),
        "skew_before": skew_before,
        "skew_after": skew_after,
        "winsor_bounds": winsor_bounds,
    }
    return out, meta


def audit_prapemrosesan(df: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """Jalankan audit lengkap Tugas 3a dan eksperimen dampak jurang nol (*zero-cliff*)."""
    # 1. Skema & integritas dasar
    null_counts = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}
    f0_mask = df["frekuensi_90h"] == 0
    krj_na = df["rata2_keranjang"].isna()
    exact_structural_missing = bool((f0_mask == krj_na).all())
    belanja_zero_when_f0 = bool((df.loc[f0_mask, "belanja_90h"] == 0.0).all())

    # 2. Deteksi bot tak-terawasi
    is_bot, bot_summary = deteksi_anomali_bot_aktivitas(df, seed=seed)
    df_norm = df.loc[~is_bot].copy()

    # 3. Verifikasi komposisional (simpleks unit)
    sum_kat = df_norm[list(PORSI_KATEGORI_COLUMNS)].sum(axis=1)
    sum_jam = df_norm[list(PORSI_JAM_COLUMNS)].sum(axis=1)
    compositional_check = {
        "porsi_kategori_sum_min": float(sum_kat.min()),
        "porsi_kategori_sum_max": float(sum_kat.max()),
        "porsi_kategori_violations_gt_0p01": int((np.abs(sum_kat - 1.0) > 0.01).sum()),
        "porsi_jam_sum_min": float(sum_jam.min()),
        "porsi_jam_sum_max": float(sum_jam.max()),
        "porsi_jam_violations_gt_0p01": int((np.abs(sum_jam - 1.0) > 0.01).sum()),
        "porsi_kategori_zero_rate_min_pct": float(
            min((df_norm[c] < 1e-3).mean() * 100.0 for c in PORSI_KATEGORI_COLUMNS)
        ),
        "porsi_kategori_zero_rate_max_pct": float(
            max((df_norm[c] < 1e-3).mean() * 100.0 for c in PORSI_KATEGORI_COLUMNS)
        ),
    }

    # 4. Verifikasi kategorikal (kota_tier vs perangkat)
    kota_stats: dict[str, dict[str, float]] = {}
    for tier, grp in df_norm.groupby("kota_tier"):
        kota_stats[str(int(tier))] = {
            "n": int(len(grp)),
            "pct": float(len(grp) / len(df_norm) * 100.0),
            "median_belanja_idr": float(grp["belanja_90h"].median()),
            "median_keranjang_idr": float(grp["rata2_keranjang"].median()),
            "median_voucher": float(grp["persen_voucher"].median()),
            "median_umur": float(grp["umur"].median()),
        }
    kw_kota_belanja = stats.kruskal(
        *[grp["belanja_90h"].to_numpy() for _, grp in df_norm.groupby("kota_tier")]
    )

    perangkat_stats: dict[str, dict[str, float]] = {}
    for dev, grp in df_norm.groupby("perangkat"):
        perangkat_stats[str(dev)] = {
            "n": int(len(grp)),
            "pct": float(len(grp) / len(df_norm) * 100.0),
            "median_belanja_idr": float(grp["belanja_90h"].median()),
            "median_keranjang_idr": float(grp["rata2_keranjang"].median()),
            "median_voucher": float(grp["persen_voucher"].median()),
            "median_cod": float(grp["persen_cod"].median()),
            "median_umur": float(grp["umur"].median()),
        }
    kw_dev_belanja = stats.kruskal(
        *[grp["belanja_90h"].to_numpy() for _, grp in df_norm.groupby("perangkat")]
    )
    mw_and_web = stats.mannwhitneyu(
        df_norm.loc[df_norm["perangkat"] == "android", "belanja_90h"].to_numpy(),
        df_norm.loc[df_norm["perangkat"] == "web", "belanja_90h"].to_numpy(),
    )

    # 5. Transformasi kanonik + bandingkan dampak jurang nol (naif vs kanonik) pada subsampel 30k
    df_canon, canon_meta = praproses_fitur_kanonik(df_norm)

    rng = np.random.default_rng(seed)
    idx_sub = rng.choice(len(df_norm), size=30_000, replace=False)
    f0_sub = (df_norm["frekuensi_90h"].to_numpy() == 0)[idx_sub]

    # Representasi naif: log1p(belanja_90h Rp) + log1p(rata2_keranjang.fillna(0) Rp)
    naive_df = pd.DataFrame(index=df_norm.index)
    for col in [
        "recency_hari",
        "frekuensi_90h",
        "belanja_90h",
        "item_per_pesanan",
        "retur_90h",
        "ulasan_90h",
        "sesi_per_minggu",
        "lama_member_bulan",
    ]:
        naive_df[col] = np.log1p(np.maximum(df_norm[col].to_numpy(dtype=np.float64), 0.0))
    naive_df["rata2_keranjang"] = np.log1p(df_norm["rata2_keranjang"].fillna(0.0).to_numpy(dtype=np.float64))
    for col in ["persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod", "umur"]:
        naive_df[col] = df_norm[col].to_numpy(dtype=np.float64)
    naive_df["is_ios"] = (df_norm["perangkat"] == "ios").astype(np.float64)
    for col in PORSI_JAM_COLUMNS[:3]:
        naive_df[col] = df_norm[col].to_numpy(dtype=np.float64)

    X_naive_sub = StandardScaler().fit_transform(naive_df.to_numpy(dtype=np.float64))[idx_sub]
    km_naive = KMeans(n_clusters=6, n_init=10, random_state=seed).fit(X_naive_sub)
    # Cek apakah ada klaster buatan yang didominasi frekuensi_90h == 0!
    naive_max_f0_share = max(
        float(f0_sub[km_naive.labels_ == cid].mean()) for cid in range(6)
    )
    naive_f0_concentration = max(
        float((km_naive.labels_[f0_sub] == cid).mean()) for cid in range(6)
    )
    naive_sil = float(silhouette_score(X_naive_sub, km_naive.labels_, sample_size=10_000, random_state=seed))

    # Representasi kanonik (tanpa jurang nol, blok inti + waktu)
    canon_core_cols = [
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
        "clr_jam_pagi",
        "clr_jam_siang",
        "clr_jam_malam",
    ]
    X_canon_sub = StandardScaler().fit_transform(df_canon[canon_core_cols].to_numpy(dtype=np.float64))[idx_sub]
    km_canon = KMeans(n_clusters=6, n_init=10, random_state=seed).fit(X_canon_sub)
    canon_max_f0_share = max(
        float(f0_sub[km_canon.labels_ == cid].mean()) for cid in range(6)
    )
    canon_f0_concentration = max(
        float((km_canon.labels_[f0_sub] == cid).mean()) for cid in range(6)
    )
    canon_sil = float(silhouette_score(X_canon_sub, km_canon.labels_, sample_size=10_000, random_state=seed))
    ari_naive_vs_canon = float(adjusted_rand_score(km_naive.labels_, km_canon.labels_))

    # Cek celah log1p(rata2_keranjang) naif antara freq==0 (nilai 0) vs freq==1 (min log1p Rp)
    krj_f1 = df_norm.loc[df_norm["frekuensi_90h"] == 1, "rata2_keranjang"].dropna()
    naive_cliff_gap = float(np.log1p(krj_f1.median()) - 0.0)
    canon_f0_med = float(df_canon.loc[df_norm["frekuensi_90h"] == 0, "log_rata2_keranjang"].median())
    canon_f1_med = float(df_canon.loc[df_norm["frekuensi_90h"] == 1, "log_rata2_keranjang"].median())

    zero_cliff_experiment = {
        "n_subsample": 30_000,
        "naive_log1p_gap_f0_to_f1_median": naive_cliff_gap,
        "canon_log_gap_f0_to_f1_median": float(abs(canon_f1_med - canon_f0_med)),
        "naive_k6_max_f0_purity_in_cluster": naive_max_f0_share,
        "naive_k6_f0_concentration_in_top_cluster": naive_f0_concentration,
        "naive_k6_silhouette": naive_sil,
        "canon_k6_max_f0_purity_in_cluster": canon_max_f0_share,
        "canon_k6_f0_concentration_in_top_cluster": canon_f0_concentration,
        "canon_k6_silhouette": canon_sil,
        "ari_naive_vs_canon_k6": ari_naive_vs_canon,
    }

    # 6. Kontinuitas perilaku frekuensi_90h == 0 vs frekuensi_90h == 1
    f0_norm = df_norm[df_norm["frekuensi_90h"] == 0]
    f1_norm = df_norm[df_norm["frekuensi_90h"] == 1]
    f0_continuity = {
        "n_f0_non_bot": int(len(f0_norm)),
        "pct_f0_non_bot": float(len(f0_norm) / len(df_norm) * 100.0),
        "f0_median_recency": float(f0_norm["recency_hari"].median()),
        "f1_median_recency": float(f1_norm["recency_hari"].median()),
        "f0_median_sesi": float(f0_norm["sesi_per_minggu"].median()),
        "f1_median_sesi": float(f1_norm["sesi_per_minggu"].median()),
        "f0_median_umur": float(f0_norm["umur"].median()),
        "f1_median_umur": float(f1_norm["umur"].median()),
        "f0_ios_pct": float((f0_norm["perangkat"] == "ios").mean() * 100.0),
        "f1_ios_pct": float((f1_norm["perangkat"] == "ios").mean() * 100.0),
    }

    return {
        "schema": {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_duplicate_ids": int(df[ID_COLUMN].duplicated().sum()),
            "n_anonymous_cols_actual": len(ANONYMOUS_N_COLUMNS),
            "n_anonymous_cols_in_kamus": 10,
            "missing_counts": null_counts,
            "exact_structural_missing_f0_eq_keranjang_na": exact_structural_missing,
            "belanja_zero_for_all_f0": belanja_zero_when_f0,
        },
        "bot_prescreening": bot_summary,
        "compositional_check": compositional_check,
        "categorical_audit": {
            "kota_tier": kota_stats,
            "kota_tier_kruskal_H": float(kw_kota_belanja.statistic),
            "kota_tier_kruskal_p": float(kw_kota_belanja.pvalue),
            "perangkat": perangkat_stats,
            "perangkat_kruskal_H": float(kw_dev_belanja.statistic),
            "perangkat_kruskal_p": float(kw_dev_belanja.pvalue),
            "android_vs_web_mannwhitney_p": float(mw_and_web.pvalue),
        },
        "canonical_transform": canon_meta,
        "f0_continuity": f0_continuity,
        "zero_cliff_experiment": zero_cliff_experiment,
    }


def buat_grafik_prapemrosesan(
    df: pd.DataFrame,
    is_bot: np.ndarray,
    df_canon: pd.DataFrame,
    audit_res: dict[str, Any],
    out_path: Path,
) -> None:
    """Buat grafik 2x2 ringkasan audit & prapemrosesan Tugas 3a."""
    atur_gaya_grafik()
    fig, axes = plt.subplots(2, 2, figsize=(LEBAR_GAMBAR, 4.9), constrained_layout=True)

    # Panel A: Pemisahan bot vs normal pada ruang aktivitas (sesi vs durasi)
    ax = axes[0, 0]
    rng = np.random.default_rng(SEED)
    norm_idx = np.flatnonzero(~is_bot)
    samp_norm = rng.choice(norm_idx, size=8_000, replace=False)
    bot_idx = np.flatnonzero(is_bot)

    ax.scatter(
        df.iloc[samp_norm]["sesi_per_minggu"],
        df.iloc[samp_norm]["durasi_sesi_menit"],
        s=3,
        alpha=0.25,
        color="#2B6CB0",
        label=f"Pelanggan normal (n={angka_id(len(norm_idx), 0)})",
        rasterized=True,
    )
    ax.scatter(
        df.iloc[bot_idx]["sesi_per_minggu"],
        df.iloc[bot_idx]["durasi_sesi_menit"],
        s=5,
        alpha=0.55,
        color="#C53030",
        label=f"Anomali bot (n={angka_id(len(bot_idx), 0)}; 1,5%)",
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(75.0, color="#718096", linestyle="--", linewidth=0.8)
    ax.axhline(1.3, color="#718096", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sesi per minggu (skala log)")
    ax.set_ylabel("Durasi sesi (menit, skala log)")
    ax.set_title("(a) Celah Mutlak Anomali Bot pada Ruang Aktivitas")
    ax.legend(loc="upper right", markerscale=2.0)

    # Panel B: Dampak jurang nol (zero-cliff) vs imputasi Ridge pada rata2_keranjang
    ax = axes[0, 1]
    df_norm = df.loc[~is_bot]
    naive_krj = np.log1p(df_norm["rata2_keranjang"].fillna(0.0).to_numpy(dtype=np.float64))
    canon_krj = df_canon["log_rata2_keranjang"].to_numpy(dtype=np.float64)
    f0_norm = (df_norm["frekuensi_90h"] == 0).to_numpy()

    bins = np.linspace(0.0, 16.0, 48)
    ax.hist(
        naive_krj[~f0_norm],
        bins=bins,
        density=True,
        alpha=0.35,
        color="#718096",
        label="Teramati (frek ≥ 1)",
    )
    gap_str = angka_id(audit_res["zero_cliff_experiment"]["naive_log1p_gap_f0_to_f1_median"], 1)
    ax.hist(
        naive_krj[f0_norm],
        bins=bins,
        density=True,
        alpha=0.75,
        color="#C53030",
        label=f"Naif isi nol (jurang {gap_str} log)",
    )
    ax.hist(
        canon_krj[f0_norm],
        bins=bins,
        density=True,
        alpha=0.70,
        color="#2F855A",
        label="Imputasi Ridge (tanpa jurang)",
    )
    ax.set_xlabel(f"Nilai transformasi {miring('log')}(rata2_keranjang)")
    ax.set_ylabel("Kepadatan")
    ax.set_title("(b) Eliminasi Jurang Nol pada Nilai Hilang Keranjang")
    sumbu_koma(ax, "xy")
    ax.legend(loc="upper center", fontsize=6.2)

    # Panel C: Kemencengan (|skewness|) sebelum vs sesudah transformasi kanonik
    ax = axes[1, 0]
    sb = audit_res["canonical_transform"]["skew_before"]
    sa = audit_res["canonical_transform"]["skew_after"]
    show_cols = [
        ("recency_hari", "Recency"),
        ("rata2_keranjang", "Keranjang"),
        ("frekuensi_90h", "Frekuensi"),
        ("lama_member_bulan", "Lama member"),
        ("item_per_pesanan", "Item/pesan"),
        ("retur_90h", "Retur 90h"),
        ("sesi_per_minggu", "Sesi/minggu"),
        ("persen_paylater", "Paylater"),
    ]
    labels = [lbl for _, lbl in show_cols]
    vals_before = [abs(sb[c]) for c, _ in show_cols]
    vals_after = [abs(sa[c]) for c, _ in show_cols]
    y_pos = np.arange(len(labels))
    h = 0.36
    b1 = ax.barh(y_pos + h / 2, vals_before, height=h, color="#CBD5E0", edgecolor="#718096", linewidth=0.4, label="Sebelum")
    b2 = ax.barh(y_pos - h / 2, vals_after, height=h, color="#2B6CB0", label="Sesudah kanonik")
    label_batang(ax, b1, desimal=1, mendatar=True, ukuran=5.8)
    label_batang(ax, b2, desimal=1, mendatar=True, ukuran=5.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals_before) * 1.22)
    ax.set_xlabel(f"Nilai mutlak kemencengan (|{miring('skewness')}|)")
    ax.set_title("(c) Penurunan Kemencengan setelah Transformasi Kanonik")
    sumbu_koma(ax, "x")
    ax.legend(loc="lower right")

    # Panel D: Audit kategorikal — perangkat (sinyal kuat iOS) vs kota_tier (noise seragam)
    ax = axes[1, 1]
    cat_aud = audit_res["categorical_audit"]
    cat_labels = ["iOS", "Android", "Web", "Tier 1", "Tier 2", "Tier 3"]
    med_spend_jt = [
        cat_aud["perangkat"]["ios"]["median_belanja_idr"] / 1e6,
        cat_aud["perangkat"]["android"]["median_belanja_idr"] / 1e6,
        cat_aud["perangkat"]["web"]["median_belanja_idr"] / 1e6,
        cat_aud["kota_tier"]["1"]["median_belanja_idr"] / 1e6,
        cat_aud["kota_tier"]["2"]["median_belanja_idr"] / 1e6,
        cat_aud["kota_tier"]["3"]["median_belanja_idr"] / 1e6,
    ]
    colors = ["#2B6CB0", "#63B3ED", "#63B3ED", "#A0AEC0", "#A0AEC0", "#A0AEC0"]
    x_pos = np.arange(len(cat_labels))
    bars = ax.bar(x_pos, med_spend_jt, width=0.58, color=colors, edgecolor="#2D3748", linewidth=0.4)
    label_batang(ax, bars, desimal=2, ukuran=6.0)
    ax.axvline(2.5, color="#718096", linestyle=":", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_labels)
    ax.set_ylim(0, max(med_spend_jt) * 1.24)
    ax.set_ylabel("Median belanja 90h (Rp juta)")
    ax.set_title("(d) Sinyal Kategorikal: Perangkat (iOS) vs Kota Tier (Seragam)")
    sumbu_koma(ax, "y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 3a: Audit & preprocessing for clustering.csv.gz.")
    parser.add_argument("--save-json", action="store_true", help="Save JSON summary to kode/results/.")
    args = parser.parse_args()

    atur_seed_global(SEED)
    siapkan_direktori()

    with catat_waktu("Task 3a: Preprocessing & Data Quality Audit"):
        df = muat_data_clustering()
        is_bot, _ = deteksi_anomali_bot_aktivitas(df, seed=SEED)
        df_norm = df.loc[~is_bot].copy()
        df_canon, _ = praproses_fitur_kanonik(df_norm)
        audit_res = audit_prapemrosesan(df, seed=SEED)

        fig_path = FIGURES_DIR / "task3_3a_preprocessing.png"
        buat_grafik_prapemrosesan(df, is_bot, df_canon, audit_res, fig_path)

        if args.tulis_json:
            tulis_json(audit_res, RESULTS_DIR / "task3_preprocess.json")

        bot = audit_res["bot_prescreening"]
        zc = audit_res["zero_cliff_experiment"]
        ct = audit_res["canonical_transform"]
        ca = audit_res["categorical_audit"]
        print(
            f"Dataset       : {len(df):,} baris x {df.shape[1]} kolom "
            f"(anomali bot pra-skala: {bot['n_bot']:,} = {bot['pct_bot']:.2f}%)"
        )
        print(
            f"Nilai hilang  : rata2_keranjang={ct['n_missing_rata2_keranjang']:,} "
            f"({ct['pct_missing_rata2_keranjang']:.2f}% non-bot, 100% frekuensi_90h==0), "
            f"Ridge R^2={ct['ridge_imputation_r2']:.4f}"
        )
        print(
            f"Jurang nol    : naif max kemurnian f0={zc['naive_k6_max_f0_purity_in_cluster']*100:.1f}% "
            f"(konsentrasi {zc['naive_k6_f0_concentration_in_top_cluster']*100:.1f}%) "
            f"-> kanonik={zc['canon_k6_max_f0_purity_in_cluster']*100:.1f}% "
            f"(ARI naif vs kanonik={zc['ari_naive_vs_canon_k6']:.4f})"
        )
        print(
            f"Kategorikal   : kota_tier Kruskal p={ca['kota_tier_kruskal_p']:.4f} (noise), "
            f"perangkat p={ca['perangkat_kruskal_p']:.2e}, android vs web p={ca['android_vs_web_mannwhitney_p']:.4f}"
        )
        print(f"Figur         : {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
