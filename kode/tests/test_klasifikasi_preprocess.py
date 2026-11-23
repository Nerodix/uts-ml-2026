"""Pengujian unit modul prapemrosesan & konstanta anti-leakage Tugas 2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.klasifikasi.preprocess import (  # noqa: E402
    ANONYMOUS_V_COLUMNS,
    CATEGORICAL_COLUMNS,
    CLEAN_BASE_FEATURES,
    CLEAN_WITH_FLAGS_FEATURES,
    DRIFT_NOISE_V_COLUMNS,
    LEAKAGE_COLUMNS,
    NAMED_NUMERIC_COLUMNS,
    NOISE_V_COLUMNS,
    QUALITY_FLAG_COLUMNS,
    SIGNAL_V_COLUMNS,
    SNAPSHOT_TIMESTAMP,
    align_categorical_dtypes,
    estimasi_kurva_maturitas_label,
    hitung_hari_sebelum_snapshot,
    tambah_flag_kualitas_data,
    urutkan_kronologis,
)


def test_schema_partitions() -> None:
    """Pastikan partisi fitur lengkap, saling lepas, dan bebas fitur leakage."""
    assert len(ANONYMOUS_V_COLUMNS) == 50
    assert len(SIGNAL_V_COLUMNS) + len(NOISE_V_COLUMNS) == 50
    assert set(SIGNAL_V_COLUMNS).isdisjoint(set(NOISE_V_COLUMNS))
    assert set(DRIFT_NOISE_V_COLUMNS).issubset(set(NOISE_V_COLUMNS))
    assert "flag_review_manual" in LEAKAGE_COLUMNS
    assert set(LEAKAGE_COLUMNS).isdisjoint(set(CLEAN_BASE_FEATURES))
    assert set(LEAKAGE_COLUMNS).isdisjoint(set(CLEAN_WITH_FLAGS_FEATURES))
    assert len(NAMED_NUMERIC_COLUMNS) == 25
    assert len(CATEGORICAL_COLUMNS) == 4
    assert len(QUALITY_FLAG_COLUMNS) == 6


def test_tambah_flag_kualitas_data() -> None:
    """Pastikan seluruh indikator kualitas baris-tunggal terhitung persis."""
    df = pd.DataFrame(
        {
            "nominal": [10_000.0, 25_000.0, 150_000.0, 1_500_000.0],
            "menit_sejak_trx_terakhir": [np.nan, 12.5, 0.5, np.nan],
            "umur_merchant_hari": [-10.0, 0.0, 15.0, 400.0],
            "persen_saldo": [0.2, 50.0, 49.9995, 1.5],
        }
    )
    out = tambah_flag_kualitas_data(df)
    assert list(out["is_trx_pertama_pengguna"]) == [1, 0, 0, 1]
    assert list(out["is_umur_merchant_negatif"]) == [1, 0, 0, 0]
    assert list(out["is_persen_saldo_capped"]) == [0, 1, 1, 0]
    assert list(out["is_nominal_bulat_100k"]) == [0, 0, 0, 1]
    assert list(out["is_nominal_bulat_50k"]) == [0, 0, 1, 1]
    assert list(out["is_nominal_kecil_25k"]) == [1, 1, 0, 0]


def test_urutkan_kronologis_dan_unseen_categorical() -> None:
    """Pastikan pengurutan waktu deterministik dan kategori unseen tidak memicu galat."""
    df_tr = pd.DataFrame(
        {
            "id_transaksi": ["T002", "T001", "T003"],
            "waktu": ["2025-05-02 10:00:00", "2025-05-01 09:00:00", "2025-05-01 09:00:00"],
            "kanal": ["qris", "kartu", "va"],
            "kategori_merchant": ["emas", "gadget", "emas"],
            "tipe_perangkat": ["android", "ios", "web"],
            "email_domain": ["gmail", "yahoo", "gmail"],
        }
    )
    df_te = pd.DataFrame(
        {
            "id_transaksi": ["T004"],
            "waktu": ["2025-09-01 08:00:00"],
            "kanal": ["kanal_baru_unseen"],
            "kategori_merchant": ["emas"],
            "tipe_perangkat": ["android"],
            "email_domain": ["gmail"],
        }
    )
    sorted_tr = urutkan_kronologis(df_tr)
    assert list(sorted_tr["id_transaksi"]) == ["T001", "T003", "T002"]

    tr_al, [te_al] = align_categorical_dtypes(df_tr, [df_te])
    assert str(tr_al["kanal"].dtype) == "category"
    assert pd.isna(te_al["kanal"].iloc[0])
    assert list(tr_al["kanal"].cat.categories) == list(te_al["kanal"].cat.categories)


def test_kurva_maturitas_sintetis() -> None:
    """Pastikan kurva maturitas mendeteksi penurunan laju fraud menjelang snapshot."""
    dates_old = pd.date_range("2025-05-01", periods=100, freq="D")
    dates_recent_0_3 = pd.date_range("2025-08-29", periods=20, freq="2h")
    df = pd.DataFrame(
        {
            "waktu_dt": list(dates_old) + list(dates_recent_0_3),
            "kanal": ["kartu"] * (len(dates_old) + len(dates_recent_0_3)),
            "is_fraud": [1 if i % 10 == 0 else 0 for i in range(len(dates_old))] + [0] * len(dates_recent_0_3),
        }
    )
    dbs = hitung_hari_sebelum_snapshot(df["waktu_dt"], SNAPSHOT_TIMESTAMP)
    assert float(dbs.min()) >= 0.0
    kurva = estimasi_kurva_maturitas_label(df)
    bin_0_3 = kurva.loc[kurva["rentang_hari"] == "0–3h", "rasio_maturitas"].iloc[0]
    assert bin_0_3 == 0.0


def main() -> None:
    test_schema_partitions()
    test_tambah_flag_kualitas_data()
    test_urutkan_kronologis_dan_unseen_categorical()
    test_kurva_maturitas_sintetis()
    print("ALL KLASIFIKASI PREPROCESS TESTS PASSED")


if __name__ == "__main__":
    main()
