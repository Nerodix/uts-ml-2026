"""Konstanta skema, deteksi bot pra-skala, dan transformasi kanonik Tugas 3a."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import CLUSTERING_PATH, SEED  # noqa: E402

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


def winsorize_kuantil(
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
        clipped, lo, hi = winsorize_kuantil(raw)
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


