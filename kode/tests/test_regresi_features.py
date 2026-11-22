"""Unit test untuk rekayasa fitur dan target encoding anti-leakage Tugas 1d."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.regresi.features import (  # noqa: E402
    HierarchicalSmoothedTargetEncoder,
    tambah_fitur_kalender_dan_domain,
)


def _dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tgl_pickup": ["2025-07-07", "2025-07-08", "2025-03-31", "2025-09-09", "2025-09-10", "2025-09-26"],
            "jam_pickup": [0, 6, 12, 18, 23, 10],
            "hub_asal": ["JKT", "JKT", "SBY", "MDN", "JKT", "BDG"],
            "pulau_tujuan": ["Jawa", "Jawa", "Papua", "Sumatera", "Jawa", "Jawa"],
            "kota_tujuan": ["KT01", "KT01", "KT99", "KT02", "KT01", "KT01"],
            "jarak_km": [100.0, 100.0, 3500.0, 400.0, 100.0, 150.0],
            "berat_kg": [1.0, 2.0, 40.0, 1.5, 1.0, 2.0],
            "volume_liter": [4.0, 8.0, 150.0, 6.0, 4.0, 8.0],
            "nilai_barang": [100_000.0, 200_000.0, 5_000_000.0, 150_000.0, 100_000.0, 120_000.0],
            "antrean_hub_asal": [1000.0, 20000.0, 500.0, 2500.0, 23000.0, 900.0],
            "hujan_asal_mm": [0.0, 10.0, 5.0, 0.0, 0.0, 2.0],
            "hujan_tujuan_mm": [0.0, 5.0, 20.0, 0.0, 0.0, 0.0],
            "alamat_skor": [0.9, 0.8, 0.5, 0.85, 0.9, 0.9],
            "f13": [-1.5, 0.0, 2.0, -0.3, 1.2, 0.5],
            "f17": [1.2, 1.5, 0.0, 1.6, 1.5, 0.1],
            "f35": [1.0, 1.28, 0.95, 1.15, 1.0, 0.94],
        }
    )


def test_calendar_and_domain_features() -> None:
    out = tambah_fitur_kalender_dan_domain(_dummy_df())
    # 7.7 dan 9.9 -> days_since_twin == 0; 7.8 dan 9.10 -> days_since_twin == 1
    assert out["days_since_twin"].tolist()[:5] == [0, 1, 7, 0, 1]
    assert out["is_twin_day0"].tolist()[:5] == [1, 0, 0, 1, 0]
    assert out["is_twin_day1_2"].tolist()[:5] == [0, 1, 0, 0, 1]
    # 2025-03-31 adalah Lebaran; tanggal lain bukan
    assert out["is_lebaran_2025"].tolist() == [0, 0, 1, 0, 0, 0]
    # 2025-09-26 masuk payday window (25-29)
    assert out["is_payday_window"].iloc[-1] == 1
    # ReLU f13_pos: negatif jadi 0, positif tetap
    assert np.allclose(out["f13_pos"].tolist(), [0.0, 0.0, 2.0, 0.0, 1.2, 0.5])


def test_hierarchical_target_encoder_fallback_unseen() -> None:
    df = tambah_fitur_kalender_dan_domain(_dummy_df())
    y = np.array([30.0, 35.0, 180.0, 55.0, 32.0, 40.0])
    enc = HierarchicalSmoothedTargetEncoder(smoothing=5.0, n_oof_splits=3, seed=42)
    tr_enc = enc.fit_transform(df, y)
    assert tr_enc["te_kota_log_mean"].notna().all()

    # Kota baru di test pada pulau yang dikenal harus fallback ke prior pulau, bukan NaN
    unseen_test = df.iloc[[0]].copy()
    unseen_test["kota_tujuan"] = "KT_BARU_999"
    unseen_test["rute_hub_kota"] = "JKT__KT_BARU_999"
    te_out = enc.transform(unseen_test)
    assert te_out["te_kota_log_mean"].notna().all()
    assert te_out["kota_tujuan_log_freq"].iloc[0] == 0.0


if __name__ == "__main__":
    test_calendar_and_domain_features()
    test_hierarchical_target_encoder_fallback_unseen()
    print("ALL REGRESI FEATURES TESTS PASSED")
