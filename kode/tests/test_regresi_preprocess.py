"""Unit test untuk modul pembersihan anti-leakage Tugas 1 (regresi/preprocess.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.regresi.preprocess import (  # noqa: E402
    BeratImputer,
    RouteDistanceImputer,
    align_categorical_dtypes,
    fix_target_unit_errors,
)


def test_fix_target_unit_errors() -> None:
    y = np.array([45.2, 693.1, 1272.0, 3600.0])
    y_clean, mask = fix_target_unit_errors(y)
    assert mask.tolist() == [False, False, True, True]
    assert np.allclose(y_clean, [45.2, 693.1, 21.2, 60.0])


def test_route_distance_imputer_hierarchy() -> None:
    train = pd.DataFrame(
        {
            "kota_asal": ["KA01", "KA01", "KA02", "KA03"],
            "kota_tujuan": ["KT01", "KT01", "KT02", "KT03"],
            "hub_asal": ["JKT", "JKT", "JKT", "SBY"],
            "pulau_tujuan": ["Jawa", "Jawa", "Jawa", "Papua"],
            "jarak_km": [120.5, 0.0, 250.0, 3100.0],
        }
    )
    test = pd.DataFrame(
        {
            "kota_asal": ["KA01", "KA99", "KA99"],
            "kota_tujuan": ["KT01", "KT88", "KT77"],
            "hub_asal": ["JKT", "JKT", "MDN"],
            "pulau_tujuan": ["Jawa", "Jawa", "Papua"],
            "jarak_km": [9999.0, 0.0, 0.0],
        }
    )
    imp = RouteDistanceImputer().fit(train)
    out = imp.transform(test, add_indicator=True)
    # Baris 0: rute eksak (KA01, KT01) -> 120.5
    # Baris 1: fallback hub+pulau (JKT, Jawa) -> median([120.5, 250.0]) = 185.25
    # Baris 2: fallback pulau (Papua) -> 3100.0
    assert np.allclose(out["jarak_km"].to_numpy(), [120.5, 185.25, 3100.0])
    assert out["jarak_km_was_corrupt"].tolist() == [1, 1, 1]


def test_berat_imputer_uses_service_density() -> None:
    train = pd.DataFrame(
        {
            "layanan": ["reguler", "kargo", "kargo"],
            "volume_liter": [4.0, 100.0, 200.0],
            "berat_kg": [1.0, 30.0, np.nan],
        }
    )
    imp = BeratImputer().fit(train)
    out = imp.transform(train, add_indicator=True)
    assert np.isclose(out.loc[2, "berat_kg"], 60.0)
    assert out["berat_kg_missing"].tolist() == [0, 0, 1]


def test_align_categorical_unseen_becomes_nan() -> None:
    tr = pd.DataFrame({"mitra_kurir": ["K01", "K02"]})
    te = pd.DataFrame({"mitra_kurir": ["K01", "K13"]})
    tr_a, [te_a] = align_categorical_dtypes(tr, [te], cat_cols=("mitra_kurir",))
    assert te_a["mitra_kurir"].iloc[0] == "K01"
    assert pd.isna(te_a["mitra_kurir"].iloc[1])


if __name__ == "__main__":
    test_fix_target_unit_errors()
    test_route_distance_imputer_hierarchy()
    test_berat_imputer_uses_service_density()
    test_align_categorical_unseen_becomes_nan()
    print("ALL REGRESI PREPROCESS TESTS PASSED")
