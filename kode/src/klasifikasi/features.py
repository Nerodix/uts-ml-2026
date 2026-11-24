"""Rekayasa fitur perilaku kausal pengguna & merchant bebas leakage masa depan (Tugas 2c).

Seluruh fitur riwayat dibangun dari aliran transaksi terurut kronologis
``(waktu_dt, id_transaksi)`` dengan jaminan kausal ketat (``t_past < t_now``):
- Fitur perilaku bebas label (pengguna & merchant) memakai pencarian biner
  tervektorisasi ``np.searchsorted`` pada kunci ``entity_code * STRIDE + waktu_sec``
  sehingga transaksi saat ini dan transaksi masa depan selalu terisolasi.
- Fitur risiko historis berbasis label memakai jeda konfirmasi 25 hari
  (``t_past <= min(t_now - 25 hari, label_cutoff)``) agar kebal terhadap
  *right-censoring* dan konsisten antara validasi maupun uji Sep–Okt.

Jalankan langsung:
    python kode/src/klasifikasi/features.py --run-ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    FRAUD_REVIEW_COST_IDR,
    KLASIFIKASI_TEST_PATH,
    KLASIFIKASI_TRAIN_PATH,
    N_JOBS,
    RESULTS_DIR,
    SEED,
    siapkan_direktori,
)
from src.klasifikasi.preprocess import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    CLEAN_WITH_FLAGS_FEATURES,
    ID_COLUMN,
    MERCHANT_ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    USER_ID_COLUMN,
    samakan_tipe_kategori,
    muat_data_klasifikasi,
    tambah_flag_kualitas_data,
)
from src.klasifikasi.validation import (  # noqa: E402
    buat_fold_waktu_matang,
    buat_split_shift_agustus,
)
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    atur_gaya_grafik,
    hitung_biaya_total_fraud,
    label_batang,
    tulis_json,
    atur_seed_global,
    sumbu_koma,
    catat_waktu,
)

# Stride detik per entitas (~317 tahun >> rentang data 10 bulan) agar jendela tidak lintas entitas
ENTITY_STRIDE = np.int64(10_000_000_000)
LAG_KONFIRMASI_DETIK = np.int64(25 * 86_400)


def _to_unix_seconds(s: pd.Series) -> np.ndarray:
    """Konversi seri waktu ke detik Unix int64 (aman untuk Pandas 2.x ns maupun Pandas 3.x us)."""
    dt = pd.to_datetime(s)
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=np.int64)

DOMAIN_INTERACTION_FEATURES: tuple[str, ...] = (
    "log_nominal",
    "log_saldo_sebelum",
    "log_jarak_lokasi_km",
    "log_menit_sejak_trx",
    "log_volume_merchant",
    "jam_sin",
    "jam_cos",
    "is_dini_hari",
    "is_akhir_pekan",
    "qris_x_bulat100k",
    "qris_x_log_nominal",
    "qris_x_rasio_nominal",
    "qris_x_persen_saldo",
    "ato_risk_sum",
    "collusive_risk_score",
    "refund_x_umur_merchant_inv",
)

USER_CAUSAL_FEATURES: tuple[str, ...] = (
    "user_past_tx_count",
    "user_past_mean_nominal",
    "user_past_std_nominal",
    "user_past_max_nominal",
    "user_nominal_zscore",
    "user_nominal_vs_max",
    "user_is_new_merchant",
    "user_merchant_repeat_count",
    "user_past_distinct_merchants",
    "user_is_new_kanal",
    "user_kanal_nominal_ratio",
    "user_burst_1j_vs_7h",
    "user_nominal_24j_vs_saldo",
)

MERCHANT_CAUSAL_FEATURES: tuple[str, ...] = (
    "merch_past_tx_count",
    "merch_days_since_first_tx",
    "merch_tx_1h",
    "merch_tx_24h",
    "merch_tx_7d",
    "merch_tx_30d",
    "merch_burst_7d_vs_30d",
    "merch_past_mean_nominal",
    "merch_nominal_vs_merch_mean",
    "merch_past_qris_share",
    "merch_past_bulat100k_share",
    "merch_past_bulat100k_7d",
    "merch_past_mean_rasio_nom",
    "merch_past_mean_persen_saldo",
    "merch_past_new_device_share",
    "merch_past_mean_v03",
    "merch_past_mean_v26",
    "merch_unique_user_ratio",
)

LAGGED_RISK_FEATURES: tuple[str, ...] = (
    "merch_lag25d_fraud_count",
    "merch_lag25d_fraud_rate_smooth",
    "user_lag25d_fraud_count",
    "user_lag25d_fraud_rate_smooth",
)

# 90 fitur perilaku kausal bebas label terpilih (terbaik pada biaya matang & QRIS shift, kebal label delay)
SELECTED_TASK2_FEATURES: tuple[str, ...] = (
    CLEAN_WITH_FLAGS_FEATURES
    + DOMAIN_INTERACTION_FEATURES
    + USER_CAUSAL_FEATURES
    + MERCHANT_CAUSAL_FEATURES
)

ALL_TASK2_FEATURES: tuple[str, ...] = SELECTED_TASK2_FEATURES + LAGGED_RISK_FEATURES


def _tambah_interaksi_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur transformasi log, siklikal kalender, dan interaksi domain baris-tunggal."""
    out = df.copy()
    w = pd.to_datetime(out["waktu_dt"] if "waktu_dt" in out.columns else out[TIME_COLUMN])
    nom = out["nominal"].to_numpy(dtype=np.float64)
    saldo = out["saldo_sebelum"].to_numpy(dtype=np.float64)
    jarak = out["jarak_lokasi_km"].to_numpy(dtype=np.float64)
    menit = out["menit_sejak_trx_terakhir"].to_numpy(dtype=np.float64)
    vol = out["volume_merchant_30h"].to_numpy(dtype=np.float64)
    jam = out["jam"].to_numpy(dtype=np.float64)
    is_qris = (out["kanal"] == "qris").to_numpy(dtype=np.float64)
    bulat100k = out["is_nominal_bulat_100k"].to_numpy(dtype=np.float64)
    rasio_nom = np.nan_to_num(out["rasio_nominal_rata2"].to_numpy(dtype=np.float64), nan=1.0)
    persen_saldo = out["persen_saldo"].to_numpy(dtype=np.float64)
    refund = out["refund_merchant_30h"].to_numpy(dtype=np.float64)
    umur_m = out["umur_merchant_hari"].to_numpy(dtype=np.float64)

    out["log_nominal"] = np.log1p(np.maximum(nom, 0.0))
    out["log_saldo_sebelum"] = np.log1p(np.maximum(saldo, 0.0))
    out["log_jarak_lokasi_km"] = np.log1p(np.maximum(jarak, 0.0))
    out["log_menit_sejak_trx"] = np.where(np.isnan(menit), np.nan, np.log1p(np.maximum(menit, 0.0)))
    out["log_volume_merchant"] = np.log1p(np.maximum(vol, 0.0))

    rad = 2.0 * np.pi * jam / 24.0
    out["jam_sin"] = np.sin(rad)
    out["jam_cos"] = np.cos(rad)
    out["is_dini_hari"] = ((jam >= 0) & (jam <= 5)).astype(np.int8)
    out["is_akhir_pekan"] = (w.dt.day_of_week >= 5).astype(np.int8)

    out["qris_x_bulat100k"] = (is_qris * bulat100k).astype(np.float32)
    out["qris_x_log_nominal"] = (is_qris * out["log_nominal"].to_numpy(dtype=np.float64)).astype(np.float32)
    out["qris_x_rasio_nominal"] = (is_qris * rasio_nom).astype(np.float32)
    out["qris_x_persen_saldo"] = (is_qris * persen_saldo).astype(np.float32)

    out["ato_risk_sum"] = (
        out["perangkat_baru"].to_numpy(dtype=np.float64)
        + out["ganti_password_7h"].to_numpy(dtype=np.float64)
        + (out["gagal_login_24j"].to_numpy(dtype=np.float64) > 0).astype(np.float64)
        + out["negara_ip_beda"].to_numpy(dtype=np.float64)
        + (jarak >= 100.0).astype(np.float64)
    ).astype(np.float32)
    out["collusive_risk_score"] = (
        np.log1p(nom / 1e6) * np.log1p(np.maximum(persen_saldo, 0.0)) * np.log1p(np.maximum(refund, 0.0))
    ).astype(np.float32)
    out["refund_x_umur_merchant_inv"] = (refund / (np.maximum(umur_m, 0.0) + 30.0)).astype(np.float32)
    return out


def _hitung_riwayat_pengguna_kausal(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Hitung 13 fitur perilaku kausal pengguna secara vektorisasi (``t_past < t_now``)."""
    n = len(df_sorted)
    u_codes, _ = pd.factorize(df_sorted[USER_ID_COLUMN], sort=False)
    ts_sec = _to_unix_seconds(df_sorted["waktu_dt"])
    keys = u_codes.astype(np.int64) * ENTITY_STRIDE + ts_sec
    order = np.argsort(keys, kind="mergesort")
    inv_order = np.empty(n, dtype=np.int64)
    inv_order[order] = np.arange(n, dtype=np.int64)

    k_s = keys[order]
    u_s = u_codes[order]
    nom_s = df_sorted["nominal"].to_numpy(dtype=np.float64)[order]

    # Indeks awal tiap grup pengguna dan batas ketat sebelum stempel waktu saat ini
    grp_start = np.searchsorted(u_s, u_s, side="left")
    strict_end = np.searchsorted(k_s, k_s, side="left")
    past_cnt = (strict_end - grp_start).astype(np.float64)

    # Welford online mean/std + running max ketat per pengguna (bebas pembatalan numerik lintas pengguna)
    mean_past = np.full(n, np.nan, dtype=np.float64)
    std_past = np.full(n, np.nan, dtype=np.float64)
    max_past = np.full(n, np.nan, dtype=np.float64)
    cur_u = -1
    cur_t = -1
    # State hingga transaksi sebelum stempel waktu saat ini (committed)
    c_n = 0
    c_mean = 0.0
    c_m2 = 0.0
    c_max = -1.0
    # Buffer transaksi pada stempel waktu saat ini (menunggu t maju)
    buf: list[float] = []

    def _flush_buf() -> None:
        nonlocal c_n, c_mean, c_m2, c_max
        for val in buf:
            c_n += 1
            delta = val - c_mean
            c_mean += delta / c_n
            c_m2 += delta * (val - c_mean)
            if val > c_max:
                c_max = val
        buf.clear()

    for i in range(n):
        ui = int(u_s[i])
        ti = int(k_s[i])
        if ui != cur_u:
            cur_u = ui
            cur_t = ti
            c_n = 0
            c_mean = 0.0
            c_m2 = 0.0
            c_max = -1.0
            buf.clear()
        elif ti > cur_t:
            _flush_buf()
            cur_t = ti

        if c_n > 0:
            mean_past[i] = c_mean
            max_past[i] = c_max
            if c_n > 1:
                std_past[i] = float(np.sqrt(max(c_m2 / c_n, 0.0)))
        buf.append(float(nom_s[i]))

    # Kebaruan pasangan (pengguna, merchant) & jumlah merchant unik sebelumnya
    um_pair = df_sorted[USER_ID_COLUMN].astype(str) + "|" + df_sorted[MERCHANT_ID_COLUMN].astype(str)
    um_codes, _ = pd.factorize(um_pair, sort=False)
    um_keys = um_codes.astype(np.int64) * ENTITY_STRIDE + ts_sec
    um_ord = np.argsort(um_keys, kind="mergesort")
    um_inv = np.empty(n, dtype=np.int64)
    um_inv[um_ord] = np.arange(n, dtype=np.int64)
    um_ks = um_keys[um_ord]
    um_cs = um_codes[um_ord]
    um_start = np.searchsorted(um_cs, um_cs, side="left")
    um_strict = np.searchsorted(um_ks, um_ks, side="left")
    um_repeat = (um_strict - um_start)[um_inv].astype(np.float32)
    is_new_merch = (um_repeat == 0).astype(np.int8)

    # Jumlah merchant unik sebelumnya per pengguna: prefix sum dari is_new_merch di urutan pengguna
    new_m_s = is_new_merch[order].astype(np.float64)
    csum_new_m = np.concatenate(([0.0], np.cumsum(new_m_s)))
    distinct_merch_past = (csum_new_m[strict_end] - csum_new_m[grp_start])[inv_order].astype(np.float32)

    # Kebaruan (pengguna, kanal) & rasio nominal terhadap rata-rata pengguna pada kanal tsb
    uk_pair = df_sorted[USER_ID_COLUMN].astype(str) + "|" + df_sorted["kanal"].astype(str)
    uk_codes, _ = pd.factorize(uk_pair, sort=False)
    uk_keys = uk_codes.astype(np.int64) * ENTITY_STRIDE + ts_sec
    uk_ord = np.argsort(uk_keys, kind="mergesort")
    uk_inv = np.empty(n, dtype=np.int64)
    uk_inv[uk_ord] = np.arange(n, dtype=np.int64)
    uk_ks = uk_keys[uk_ord]
    uk_cs = uk_codes[uk_ord]
    uk_start = np.searchsorted(uk_cs, uk_cs, side="left")
    uk_strict = np.searchsorted(uk_ks, uk_ks, side="left")
    # Jumlah transaksi sebelumnya pada pasangan (pengguna, kanal) & rata-rata nominal per grup
    uk_cnt = (uk_strict - uk_start).astype(np.float64)
    uk_nom_s = df_sorted["nominal"].to_numpy(dtype=np.float64)[uk_ord]
    uk_gsum = pd.Series(uk_nom_s).groupby(uk_cs, sort=False).cumsum().to_numpy(dtype=np.float64)
    uk_sum_past = np.where(uk_cnt > 0, uk_gsum[np.maximum(uk_strict - 1, 0)], 0.0)
    uk_mean = np.where(uk_cnt > 0, uk_sum_past / np.maximum(uk_cnt, 1.0), np.nan)
    uk_ratio = np.where(uk_cnt > 0, uk_nom_s / np.maximum(uk_mean, 1.0), np.nan)[uk_inv]
    is_new_kanal = (uk_cnt[uk_inv] == 0).astype(np.int8)

    # Kembalikan ke urutan df_sorted
    past_cnt_out = past_cnt[inv_order]
    mean_past_out = mean_past[inv_order]
    std_past_out = std_past[inv_order]
    max_past_out = max_past[inv_order]
    nom_raw = df_sorted["nominal"].to_numpy(dtype=np.float64)

    res = pd.DataFrame(index=df_sorted.index)
    res["user_past_tx_count"] = past_cnt_out.astype(np.float32)
    res["user_past_mean_nominal"] = mean_past_out.astype(np.float32)
    res["user_past_std_nominal"] = std_past_out.astype(np.float32)
    res["user_past_max_nominal"] = max_past_out.astype(np.float32)
    res["user_nominal_zscore"] = np.where(
        np.isnan(mean_past_out),
        np.nan,
        (nom_raw - mean_past_out) / (np.nan_to_num(std_past_out, nan=0.0) + 10_000.0),
    ).astype(np.float32)
    res["user_nominal_vs_max"] = np.where(
        np.isnan(max_past_out), np.nan, nom_raw / np.maximum(max_past_out, 1.0)
    ).astype(np.float32)
    res["user_is_new_merchant"] = is_new_merch
    res["user_merchant_repeat_count"] = um_repeat
    res["user_past_distinct_merchants"] = distinct_merch_past
    res["user_is_new_kanal"] = is_new_kanal
    res["user_kanal_nominal_ratio"] = uk_ratio.astype(np.float32)
    res["user_burst_1j_vs_7h"] = (
        df_sorted["trx_1j"].to_numpy(dtype=np.float64)
        / (df_sorted["trx_7h"].to_numpy(dtype=np.float64) + 1.0)
    ).astype(np.float32)
    res["user_nominal_24j_vs_saldo"] = (
        df_sorted["nominal_24j"].to_numpy(dtype=np.float64)
        / (df_sorted["saldo_sebelum"].to_numpy(dtype=np.float64) + 1.0)
    ).astype(np.float32)
    return res


def _hitung_riwayat_merchant_kausal(df_sorted: pd.DataFrame, is_new_merch_user: np.ndarray) -> pd.DataFrame:
    """Hitung 18 fitur perilaku kausal merchant secara vektorisasi (``t_past < t_now``)."""
    n = len(df_sorted)
    m_codes, _ = pd.factorize(df_sorted[MERCHANT_ID_COLUMN], sort=False)
    ts_sec = _to_unix_seconds(df_sorted["waktu_dt"])
    keys = m_codes.astype(np.int64) * ENTITY_STRIDE + ts_sec
    order = np.argsort(keys, kind="mergesort")
    inv_order = np.empty(n, dtype=np.int64)
    inv_order[order] = np.arange(n, dtype=np.int64)

    k_s = keys[order]
    m_s = m_codes[order]
    ts_s = ts_sec[order]

    grp_start = np.searchsorted(m_s, m_s, side="left")
    strict_end = np.searchsorted(k_s, k_s, side="left")
    past_cnt = (strict_end - grp_start).astype(np.float64)

    # Lama hari sejak transaksi pertama merchant terlihat di log (tanpa bocor masa depan)
    first_ts = ts_s[grp_start]
    days_since_first = np.where(past_cnt > 0, (ts_s - first_ts) / 86_400.0, 0.0)

    # Jendela kecepatan transaksi merchant (1 jam, 24 jam, 7 hari, 30 hari)
    l_1h = np.searchsorted(k_s, k_s - np.int64(3_600), side="left")
    l_24h = np.searchsorted(k_s, k_s - np.int64(86_400), side="left")
    l_7d = np.searchsorted(k_s, k_s - np.int64(7 * 86_400), side="left")
    l_30d = np.searchsorted(k_s, k_s - np.int64(30 * 86_400), side="left")

    tx_1h = (strict_end - l_1h).astype(np.float64)
    tx_24h = (strict_end - l_24h).astype(np.float64)
    tx_7d = (strict_end - l_7d).astype(np.float64)
    tx_30d = (strict_end - l_30d).astype(np.float64)
    burst_7_30 = tx_7d / (tx_30d / 4.2857 + 1.0)

    def _past_mean(vals: np.ndarray) -> np.ndarray:
        v_s = vals[order]
        valid = (~np.isnan(v_s)).astype(np.float64)
        v_clean = np.nan_to_num(v_s, nan=0.0)
        g_v = pd.Series(v_clean).groupby(m_s, sort=False).cumsum().to_numpy(dtype=np.float64)
        g_c = pd.Series(valid).groupby(m_s, sort=False).cumsum().to_numpy(dtype=np.float64)
        prev_idx = np.maximum(strict_end - 1, 0)
        has_past = strict_end > grp_start
        cnt = np.where(has_past, g_c[prev_idx], 0.0)
        tot = np.where(has_past, g_v[prev_idx], 0.0)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1.0), np.nan)

    nom = df_sorted["nominal"].to_numpy(dtype=np.float64)
    is_qris = (df_sorted["kanal"] == "qris").to_numpy(dtype=np.float64)
    bulat100k = df_sorted["is_nominal_bulat_100k"].to_numpy(dtype=np.float64)
    rasio_nom = df_sorted["rasio_nominal_rata2"].to_numpy(dtype=np.float64)
    persen_saldo = df_sorted["persen_saldo"].to_numpy(dtype=np.float64)
    perangkat_baru = df_sorted["perangkat_baru"].to_numpy(dtype=np.float64)
    v03 = df_sorted["v03"].to_numpy(dtype=np.float64)
    v26 = df_sorted["v26"].to_numpy(dtype=np.float64)

    m_mean_nom = _past_mean(nom)
    m_qris_share = _past_mean(is_qris)
    m_bulat_share = _past_mean(bulat100k)
    m_mean_rasio = _past_mean(rasio_nom)
    m_mean_psaldo = _past_mean(persen_saldo)
    m_new_dev_share = _past_mean(perangkat_baru)
    m_mean_v03 = _past_mean(v03)
    m_mean_v26 = _past_mean(v26)

    # Jumlah transaksi bulat 100rb di merchant dalam 7 hari terakhir (detektor shell QRIS)
    cs_bulat = np.concatenate(([0.0], np.cumsum(bulat100k[order])))
    m_bulat_7d = cs_bulat[strict_end] - cs_bulat[l_7d]

    # Rasio pengguna unik terhadap total transaksi historis merchant
    cs_new_u = np.concatenate(([0.0], np.cumsum(is_new_merch_user[order].astype(np.float64))))
    uniq_u_past = cs_new_u[strict_end] - cs_new_u[grp_start]
    uniq_u_ratio = np.where(past_cnt > 0, uniq_u_past / np.maximum(past_cnt, 1.0), np.nan)

    res = pd.DataFrame(index=df_sorted.index)
    res["merch_past_tx_count"] = past_cnt[inv_order].astype(np.float32)
    res["merch_days_since_first_tx"] = days_since_first[inv_order].astype(np.float32)
    res["merch_tx_1h"] = tx_1h[inv_order].astype(np.float32)
    res["merch_tx_24h"] = tx_24h[inv_order].astype(np.float32)
    res["merch_tx_7d"] = tx_7d[inv_order].astype(np.float32)
    res["merch_tx_30d"] = tx_30d[inv_order].astype(np.float32)
    res["merch_burst_7d_vs_30d"] = burst_7_30[inv_order].astype(np.float32)
    res["merch_past_mean_nominal"] = m_mean_nom[inv_order].astype(np.float32)
    res["merch_nominal_vs_merch_mean"] = np.where(
        np.isnan(m_mean_nom[inv_order]), np.nan, nom / np.maximum(m_mean_nom[inv_order], 1.0)
    ).astype(np.float32)
    res["merch_past_qris_share"] = m_qris_share[inv_order].astype(np.float32)
    res["merch_past_bulat100k_share"] = m_bulat_share[inv_order].astype(np.float32)
    res["merch_past_bulat100k_7d"] = m_bulat_7d[inv_order].astype(np.float32)
    res["merch_past_mean_rasio_nom"] = m_mean_rasio[inv_order].astype(np.float32)
    res["merch_past_mean_persen_saldo"] = m_mean_psaldo[inv_order].astype(np.float32)
    res["merch_past_new_device_share"] = m_new_dev_share[inv_order].astype(np.float32)
    res["merch_past_mean_v03"] = m_mean_v03[inv_order].astype(np.float32)
    res["merch_past_mean_v26"] = m_mean_v26[inv_order].astype(np.float32)
    res["merch_unique_user_ratio"] = uniq_u_ratio[inv_order].astype(np.float32)
    return res


def bangun_fitur_bebas_label(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Bangun seluruh fitur kausal bebas label (kualitas, interaksi, riwayat user & merchant).

    Menggabungkan aliran transaksi ``train`` dan ``test`` **tanpa kolom label**,
    mengurutkan stabil berdasarkan ``(waktu_dt, id_transaksi)``, menghitung seluruh
    agregasi perilaku ketat sebelum ``t_now``, lalu memetakan kembali ke urutan baris
    asli masing-masing DataFrame.

    Parameters
    ----------
    train_raw : pd.DataFrame
        Data latih klasifikasi mentah.
    test_raw : pd.DataFrame or None, default=None
        Data uji klasifikasi mentah (opsional).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        Pasangan ``(train_feat, test_feat)`` berurutan baris identik dengan input asli.
    """
    tr = tambah_flag_kualitas_data(train_raw)
    if "waktu_dt" not in tr.columns:
        tr["waktu_dt"] = pd.to_datetime(tr[TIME_COLUMN])
    tr["_is_train"] = np.int8(1)
    tr["_orig_pos"] = np.arange(len(tr), dtype=np.int64)

    if test_raw is not None:
        te = tambah_flag_kualitas_data(test_raw)
        if "waktu_dt" not in te.columns:
            te["waktu_dt"] = pd.to_datetime(te[TIME_COLUMN])
        te["_is_train"] = np.int8(0)
        te["_orig_pos"] = np.arange(len(te), dtype=np.int64)
        # Drop target sebelum concat agar dijamin tidak mungkin tersentuh
        tr_nolabel = tr.drop(columns=[TARGET_COLUMN], errors="ignore")
        gabung = pd.concat([tr_nolabel, te], ignore_index=True)
    else:
        gabung = tr.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

    gabung = _tambah_interaksi_domain(gabung)
    gabung_sorted = gabung.sort_values(["waktu_dt", ID_COLUMN], kind="mergesort")

    u_feats = _hitung_riwayat_pengguna_kausal(gabung_sorted)
    m_feats = _hitung_riwayat_merchant_kausal(
        gabung_sorted, u_feats["user_is_new_merchant"].to_numpy(dtype=np.int8)
    )
    gabung_full = pd.concat([gabung_sorted, u_feats, m_feats], axis=1)

    tr_mask = gabung_full["_is_train"].to_numpy() == 1
    tr_out = (
        gabung_full.loc[tr_mask]
        .sort_values("_orig_pos", kind="mergesort")
        .drop(columns=["_is_train", "_orig_pos"])
        .reset_index(drop=True)
    )
    if TARGET_COLUMN in train_raw.columns:
        tr_out[TARGET_COLUMN] = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)

    if test_raw is None:
        return tr_out, None

    te_out = (
        gabung_full.loc[~tr_mask]
        .sort_values("_orig_pos", kind="mergesort")
        .drop(columns=["_is_train", "_orig_pos"])
        .reset_index(drop=True)
    )
    return tr_out, te_out


def tambah_fitur_risiko_lag25h(
    train_feat: pd.DataFrame,
    eval_feat: pd.DataFrame | None,
    label_cutoff: str | pd.Timestamp,
    lag_hari: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Tambahkan fitur risiko fraud historis merchant & pengguna dengan jeda konfirmasi 25 hari.

    Untuk setiap transaksi pada waktu ``t``, hanya menghitung transaksi dari
    ``train_feat`` yang:
    1. Terjadi pada waktu ``t_past <= min(t - lag_hari, label_cutoff)``, sehingga
       100% bebas *future leakage* sekaligus bebas *right-censoring* di dalam fold.
    2. Menggunakan *empirical Bayes smoothing* menuju laju fraud prior latih
       (``m_merchant = 10``, ``m_user = 5``).

    Parameters
    ----------
    train_feat : pd.DataFrame
        Data latih berfitur bebas-label (harus memuat ``is_fraud``).
    eval_feat : pd.DataFrame or None
        Data validasi atau uji yang ingin diperkaya memakai label <= ``label_cutoff``.
    label_cutoff : str or pd.Timestamp
        Tanggal batas maksimum label latih yang boleh terlihat (mis. ``fold.train_end``).
    lag_hari : int, default=25
        Jeda konfirmasi minimum (hari) sebelum label boleh masuk ke fitur historis.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        Salinan ``(train_out, eval_out)`` dengan 4 kolom ``LAGGED_RISK_FEATURES``.
    """
    cutoff_ts = pd.Timestamp(label_cutoff)
    if cutoff_ts.hour == 0 and cutoff_ts.minute == 0 and cutoff_ts.second == 0:
        cutoff_ts = cutoff_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    cutoff_sec = np.int64(int((cutoff_ts - pd.Timestamp("1970-01-01")).total_seconds()))
    lag_sec = np.int64(lag_hari * 86_400)

    tr_out = train_feat.copy()
    ev_out = eval_feat.copy() if eval_feat is not None else None

    w_tr_sec = _to_unix_seconds(tr_out["waktu_dt"])
    y_tr = tr_out[TARGET_COLUMN].to_numpy(dtype=np.float64)
    valid_lbl = w_tr_sec <= cutoff_sec
    prior = float(y_tr[valid_lbl].mean()) if valid_lbl.any() else 0.0068

    if ev_out is not None:
        w_ev_sec = _to_unix_seconds(ev_out["waktu_dt"])

    for col_ent, prefix, m_smooth in (
        (MERCHANT_ID_COLUMN, "merch", 10.0),
        (USER_ID_COLUMN, "user", 5.0),
    ):
        # Ambil hanya baris latih dengan label diketahui (<= cutoff_sec)
        src_ent = tr_out.loc[valid_lbl, col_ent].astype(str).to_numpy()
        src_ts = w_tr_sec[valid_lbl]
        src_y = y_tr[valid_lbl]

        # Faktorisasi kode entitas bersama antara sumber label, train_out, dan eval_out
        tr_ent = tr_out[col_ent].astype(str).to_numpy()
        if ev_out is not None:
            ev_ent = ev_out[col_ent].astype(str).to_numpy()
            all_ent = np.concatenate([src_ent, tr_ent, ev_ent])
        else:
            all_ent = np.concatenate([src_ent, tr_ent])
        codes_all, _ = pd.factorize(all_ent, sort=False)
        n_src = len(src_ent)
        n_tr = len(tr_ent)
        c_src = codes_all[:n_src].astype(np.int64)
        c_tr = codes_all[n_src : n_src + n_tr].astype(np.int64)

        src_keys = c_src * ENTITY_STRIDE + src_ts
        ord_s = np.argsort(src_keys, kind="mergesort")
        k_s = src_keys[ord_s]
        c_s = c_src[ord_s]
        y_s = src_y[ord_s]
        cs_y = np.concatenate(([0.0], np.cumsum(y_s)))

        def _query(c_q: np.ndarray, ts_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            eff_ts = np.minimum(ts_q - lag_sec, cutoff_sec)
            q_start = np.searchsorted(c_s, c_q, side="left")
            q_end = np.searchsorted(k_s, c_q * ENTITY_STRIDE + eff_ts, side="right")
            cnt = np.maximum(q_end - q_start, 0).astype(np.float64)
            fsum = np.where(cnt > 0, cs_y[q_end] - cs_y[q_start], 0.0)
            rate = (fsum + m_smooth * prior) / (cnt + m_smooth)
            return fsum.astype(np.float32), rate.astype(np.float32)

        f_tr, r_tr = _query(c_tr, w_tr_sec)
        tr_out[f"{prefix}_lag25d_fraud_count"] = f_tr
        tr_out[f"{prefix}_lag25d_fraud_rate_smooth"] = r_tr

        if ev_out is not None:
            c_ev = codes_all[n_src + n_tr :].astype(np.int64)
            f_ev, r_ev = _query(c_ev, w_ev_sec)
            ev_out[f"{prefix}_lag25d_fraud_count"] = f_ev
            ev_out[f"{prefix}_lag25d_fraud_rate_smooth"] = r_ev

    return tr_out, ev_out


def verifikasi_kausalitas_fitur(train_raw: pd.DataFrame) -> None:
    """Pastikan mutasi seluruh transaksi masa depan tidak mengubah 1 angka pun di masa lalu."""
    sample = train_raw.iloc[:30_000].copy()
    cutoff = pd.Timestamp("2025-05-01 00:00:00")
    past_mask = pd.to_datetime(sample["waktu_dt"]) < cutoff

    f_asli, _ = bangun_fitur_bebas_label(sample, None)
    f_asli, _ = tambah_fitur_risiko_lag25h(f_asli, None, label_cutoff="2025-04-30")

    sample_mut = sample.copy()
    fut = ~past_mask
    sample_mut.loc[fut, "nominal"] = sample_mut.loc[fut, "nominal"] * 100.0 + 999_000.0
    sample_mut.loc[fut, "kanal"] = "qris"
    sample_mut.loc[fut, "v26"] = 99.0
    sample_mut.loc[fut, TARGET_COLUMN] = 1

    f_mut, _ = bangun_fitur_bebas_label(sample_mut, None)
    f_mut, _ = tambah_fitur_risiko_lag25h(f_mut, None, label_cutoff="2025-04-30")

    cek_cols = list(USER_CAUSAL_FEATURES + MERCHANT_CAUSAL_FEATURES + LAGGED_RISK_FEATURES)
    for col in cek_cols:
        a = f_asli.loc[past_mask, col].to_numpy(dtype=np.float64)
        b = f_mut.loc[past_mask, col].to_numpy(dtype=np.float64)
        assert np.allclose(a, b, rtol=0.0, atol=1e-7, equal_nan=True), f"Leakage terdeteksi di {col}!"


def _evaluasi_konfigurasi_fitur(
    train_base: pd.DataFrame,
    fitur: list[str],
    pakai_lag_risiko: bool,
    seed: int = SEED,
) -> tuple[dict[str, Any], pd.Series]:
    """Uji konfigurasi fitur pada 3-Fold Expanding-Window Time CV Matang + QRIS Agt 11–21."""
    folds = buat_fold_waktu_matang(train_base)
    params = {
        "objective": "binary",
        "n_estimators": 350,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }
    pr_list, roc_list, cost_list = [], [], []
    gain_total = np.zeros(len(fitur), dtype=np.float64)

    for f in folds:
        tr_f = train_base.iloc[f.train_idx]
        va_f = train_base.iloc[f.val_idx]
        if pakai_lag_risiko:
            tr_f, va_f = tambah_fitur_risiko_lag25h(tr_f, va_f, label_cutoff=f.train_end)
        tr_al, [va_al] = samakan_tipe_kategori(tr_f, [va_f], cat_cols=CATEGORICAL_COLUMNS)
        clf = lgb.LGBMClassifier(**params)
        clf.fit(tr_al[fitur], tr_al[TARGET_COLUMN].to_numpy(dtype=np.int8))
        prob = np.clip(clf.predict_proba(va_al[fitur])[:, 1], 1e-7, 1.0 - 1e-7)
        y_va = va_al[TARGET_COLUMN].to_numpy(dtype=np.int8)
        nom_va = va_al["nominal"].to_numpy(dtype=np.float64)
        flg = (prob * nom_va > FRAUD_REVIEW_COST_IDR).astype(np.int8)
        pr_list.append(float(average_precision_score(y_va, prob)))
        roc_list.append(float(roc_auc_score(y_va, prob)))
        cost_list.append(float(hitung_biaya_total_fraud(y_va, flg, nom_va) / len(y_va) * 10_000.0))
        gain_total += clf.booster_.feature_importance(importance_type="gain")

    # Cek tambahan pada gelombang QRIS Agustus (latih Jan–Jul + Agt 1–10 -> eval Agt 11–21 QRIS)
    sp = buat_split_shift_agustus(train_base)
    tr_s = train_base.iloc[np.concatenate([sp["jan_jul_idx"], sp["aug_adapt_idx"]])]
    va_s = train_base.iloc[sp["aug_val_idx"]]
    va_sq = va_s[va_s["kanal"] == "qris"]
    if pakai_lag_risiko:
        tr_s, va_sq = tambah_fitur_risiko_lag25h(tr_s, va_sq, label_cutoff="2025-08-10")
    tr_sal, [va_sal] = samakan_tipe_kategori(tr_s, [va_sq], cat_cols=CATEGORICAL_COLUMNS)
    clf_s = lgb.LGBMClassifier(**params)
    clf_s.fit(tr_sal[fitur], tr_sal[TARGET_COLUMN].to_numpy(dtype=np.int8))
    p_sq = clf_s.predict_proba(va_sal[fitur])[:, 1]
    pr_qris_shift = float(average_precision_score(va_sal[TARGET_COLUMN].to_numpy(dtype=np.int8), p_sq))

    met = {
        "n_features": len(fitur),
        "mean_pr_auc": float(np.mean(pr_list)),
        "std_pr_auc": float(np.std(pr_list, ddof=1)),
        "mean_roc_auc": float(np.mean(roc_list)),
        "mean_cost_per_10k_idr": float(np.mean(cost_list)),
        "qris_shift_pr_auc": pr_qris_shift,
    }
    return met, pd.Series(gain_total / len(folds), index=fitur)


def simpan_plot_fitur_perilaku(
    ablasi: list[dict[str, Any]],
    top_gain: list[dict[str, Any]],
    output_path: Path = FIGURES_DIR / "task2_2c_behavioral_features.png",
) -> Path:
    """Gambar figur 2 panel ablasi fitur perilaku kausal 2c & top fitur perilaku."""
    atur_gaya_grafik()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.75), gridspec_kw={"width_ratios": [1.08, 1.0]})

    # Panel A: Progresi PR-AUC (Time CV Matang & QRIS Agt 11–21)
    ax0 = axes[0]
    lbls = [r["label_pendek"] for r in ablasi]
    pr_mat = [r["mean_pr_auc"] for r in ablasi]
    pr_qris = [r["qris_shift_pr_auc"] for r in ablasi]
    x = np.arange(len(lbls))
    w = 0.36
    b1 = ax0.bar(x - w / 2, pr_mat, width=w, color="#1f77b4", label="Time CV matang (Mei–Jul)")
    b2 = ax0.bar(x + w / 2, pr_qris, width=w, color="#ff7f0e", label="QRIS Agt 11–21 (shift)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(lbls, fontsize=6.0)
    ax0.set_ylim(0, max(max(pr_mat), max(pr_qris)) * 1.24)
    ax0.set_ylabel("PR-AUC")
    ax0.set_title("(a) Ablasi bertahap fitur perilaku kausal")
    sumbu_koma(ax0, "y")
    label_batang(ax0, b1, desimal=3, ukuran=5.4, putar=True)
    label_batang(ax0, b2, desimal=3, ukuran=5.4, putar=True)
    ax0.legend(loc="lower right", fontsize=5.8)

    # Panel B: Top 10 fitur rekayasa perilaku berdasarkan rerata gain LightGBM
    ax1 = axes[1]
    top10 = list(reversed(top_gain[:10]))
    y_pos = np.arange(len(top10))
    vals = [r["gain_ribu"] for r in top10]
    cols = ["#e6550d" if r["fitur"].startswith("merch_") else ("#2ca02c" if r["fitur"].startswith("user_") else "#6baed6") for r in top10]
    ax1.barh(y_pos, vals, color=cols, height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r["fitur"] for r in top10], fontsize=6.0)
    ax1.set_xlabel("Rerata gain LightGBM (ribuan)")
    ax1.set_title("(b) 10 fitur perilaku kausal terpenting")
    sumbu_koma(ax1, "x")

    fig.tight_layout(w_pad=1.5)
    siapkan_direktori()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def jalankan_ablasi_fitur(save_json_flag: bool = True) -> dict[str, Any]:
    """Jalankan verifikasi anti-leakage kausalitas dan ablasi 5 tahap fitur Tugas 2c."""
    atur_seed_global(SEED)
    siapkan_direktori()

    with catat_waktu("Muat data & verifikasi kausalitas ketat (Tugas 2c)"):
        train_raw, _ = muat_data_klasifikasi(KLASIFIKASI_TRAIN_PATH, KLASIFIKASI_TEST_PATH)
        verifikasi_kausalitas_fitur(train_raw)
        print("  Verifikasi kausalitas LULUS: mutasi masa depan mengubah fitur masa lalu sebesar 0,0.")
        train_feat, _ = bangun_fitur_bebas_label(train_raw, None)

    tahapan = [
        ("dasar", "Dasar\n(2a–2b)", list(CLEAN_WITH_FLAGS_FEATURES), False),
        ("plus_domain", "+Domain\n& jam", list(CLEAN_WITH_FLAGS_FEATURES + DOMAIN_INTERACTION_FEATURES), False),
        ("plus_user", "+Riwayat\npengguna", list(CLEAN_WITH_FLAGS_FEATURES + DOMAIN_INTERACTION_FEATURES + USER_CAUSAL_FEATURES), False),
        ("plus_merchant", "+Riwayat\nmerchant", list(CLEAN_WITH_FLAGS_FEATURES + DOMAIN_INTERACTION_FEATURES + USER_CAUSAL_FEATURES + MERCHANT_CAUSAL_FEATURES), False),
        ("lengkap_2c", "+Risiko\nlag-25h", list(ALL_TASK2_FEATURES), True),
    ]

    ablasi: list[dict[str, Any]] = []
    gain_lengkap = pd.Series(dtype=float)
    for kode, lbl, cols, lag_flag in tahapan:
        with catat_waktu(f"2c Ablasi: {kode} ({len(cols)} fitur)"):
            met, gain = _evaluasi_konfigurasi_fitur(train_feat, cols, lag_flag, seed=SEED)
            met["tahap"] = kode
            met["label_pendek"] = lbl
            ablasi.append(met)
            print(
                f"  PR-AUC matang={met['mean_pr_auc']:.4f} ± {met['std_pr_auc']:.4f} | "
                f"Biaya/10k={met['mean_cost_per_10k_idr']:,.0f} Rp | QRIS Agt={met['qris_shift_pr_auc']:.4f}"
            )
            if kode == "lengkap_2c":
                gain_lengkap = gain

    engineered = set(DOMAIN_INTERACTION_FEATURES + USER_CAUSAL_FEATURES + MERCHANT_CAUSAL_FEATURES + LAGGED_RISK_FEATURES)
    top_eng = gain_lengkap[gain_lengkap.index.isin(engineered)].sort_values(ascending=False).head(15)
    top_gain = [{"fitur": str(k), "gain_ribu": float(v / 1000.0)} for k, v in top_eng.items()]

    simpan_plot_fitur_perilaku(ablasi, top_gain)
    hasil = {
        "kausalitas_terverifikasi": True,
        "tahap_terpilih": "plus_merchant",
        "jumlah_fitur_terpilih": len(SELECTED_TASK2_FEATURES),
        "alasan_pemilihan": "Himpunan 90 fitur kausal bebas-label (plus_merchant) memberi biaya terendah "
        "dan PR-AUC tertinggi pada gelombang QRIS baru, sekaligus kebal terhadap jeda konfirmasi label "
        "pada merchant baru di data uji Sep–Okt.",
        "rincian_kelompok": {
            "dasar_bersih_dan_flag": len(CLEAN_WITH_FLAGS_FEATURES),
            "interaksi_domain": len(DOMAIN_INTERACTION_FEATURES),
            "riwayat_kausal_pengguna": len(USER_CAUSAL_FEATURES),
            "riwayat_kausal_merchant": len(MERCHANT_CAUSAL_FEATURES),
            "risiko_historis_lag25h_diuji": len(LAGGED_RISK_FEATURES),
        },
        "ablasi": [{k: v for k, v in r.items() if k != "label_pendek"} for r in ablasi],
        "top_fitur_perilaku": top_gain,
    }
    if save_json_flag:
        tulis_json(hasil, RESULTS_DIR / "task2_features.json")
        print("[SAVED] task2_features.json")
    return hasil


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2c causal behavioral feature engineering & ablation.")
    parser.add_argument("--run-ablation", action="store_true", help="Run 5-stage feature ablation on Time CV.")
    parser.add_argument("--no-save-json", action="store_true", help="Skip saving task2_features.json.")
    args = parser.parse_args()
    if args.run_ablation:
        jalankan_ablasi_fitur(save_json_flag=not args.no_save_json)
    else:
        jalankan_ablasi_fitur(save_json_flag=not args.no_save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
