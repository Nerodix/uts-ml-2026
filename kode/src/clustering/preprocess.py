"""Konstanta skema dan pra-deteksi anomali bot/outlier Tugas 3a (Segmentasi Pelanggan)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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


