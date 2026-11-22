"""Unit test untuk kalibrasi Conformal Prediction Tugas 1e."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.regresi.calibration import (  # noqa: E402
    AdditiveCQR,
    GroupMultiplicativeCQR,
    MultiplicativeCQR,
    conformal_quantile,
)


def test_conformal_quantile_finite_sample_guarantee() -> None:
    rng = np.random.default_rng(42)
    y = rng.lognormal(mean=4.0, sigma=0.4, size=2000)
    p50 = np.full_like(y, np.median(y))
    # Sengaja buat p90 mentah under-cover (~80%)
    p90_raw = np.full_like(y, np.quantile(y, 0.80))

    y_cal, y_ev = y[:1000], y[1000:]
    p50_ev, p90_cal, p90_ev = p50[1000:], p90_raw[:1000], p90_raw[1000:]

    add = AdditiveCQR(alpha=0.10).fit(y_cal, p90_cal)
    p90_add = add.transform(p50_ev, p90_ev)
    assert (p90_add >= p50_ev).all()
    assert 0.87 <= np.mean(y_ev <= p90_add) <= 0.93

    mul = MultiplicativeCQR(alpha=0.10).fit(y_cal, p90_cal)
    p90_mul = mul.transform(p50_ev, p90_ev)
    assert (p90_mul >= p50_ev).all()
    assert 0.87 <= np.mean(y_ev <= p90_mul) <= 0.93

    grp = np.where(np.arange(2000) % 2 == 0, "reguler", "kargo")
    gc = GroupMultiplicativeCQR(alpha=0.10).fit(y_cal, p90_cal, grp[:1000])
    p90_g = gc.transform(p50_ev, p90_ev, grp[1000:])
    assert (p90_g >= p50_ev).all()
    assert conformal_quantile(np.arange(1, 101, dtype=float), alpha=0.10) == 91.0


if __name__ == "__main__":
    test_conformal_quantile_finite_sample_guarantee()
    print("ALL REGRESI CALIBRATION TESTS PASSED")
