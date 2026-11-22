"""Verifikasi unit sederhana untuk metrik UTS dan validator output."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.utils import fraud_total_cost, p90_coverage, pinball_loss  # noqa: E402
from src.validate_outputs import (  # noqa: E402
    validate_clustering,
    validate_klasifikasi,
    validate_regresi,
)


def test_metrics() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([15.0, 20.0, 25.0, 40.0])
    # q=0.5: 0.5 * mean(|y - yhat|) = 0.5 * (5 + 0 + 5 + 0) / 4 = 1.25
    assert np.isclose(pinball_loss(y_true, y_pred, 0.5), 1.25)
    # q=0.9: diffs = [-5, 0, +5, 0] -> losses = [0.1*5, 0, 0.9*5, 0] = [0.5, 0, 4.5, 0] -> mean 1.25
    assert np.isclose(pinball_loss(y_true, y_pred, 0.9), 1.25)
    assert np.isclose(p90_coverage(y_true, np.array([15.0, 20.0, 25.0, 50.0])), 0.75)

    # Fraud cost: missed fraud nominal (500k) + 2 flagged * 25k = 550k
    yt = np.array([1, 1, 0, 0])
    fl = np.array([1, 0, 1, 0])
    nom = np.array([200_000.0, 500_000.0, 100_000.0, 50_000.0])
    assert np.isclose(fraud_total_cost(yt, fl, nom, review_cost=25_000.0), 550_000.0)


def test_validators_catch_contract_violations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Regresi: p90 < p50 harus gagal
        ref_r = d / "ref_r.csv"
        pd.DataFrame({"id_paket": ["P1", "P2"]}).to_csv(ref_r, index=False)
        ok_r = d / "ok_r.csv"
        pd.DataFrame({"id_paket": ["P1", "P2"], "p50": [10.0, 20.0], "p90": [15.0, 20.0]}).to_csv(ok_r, index=False)
        assert validate_regresi(ok_r, ref_r) == []
        bad_r = d / "bad_r.csv"
        pd.DataFrame({"id_paket": ["P1", "P2"], "p50": [10.0, 25.0], "p90": [15.0, 20.0]}).to_csv(bad_r, index=False)
        assert len(validate_regresi(bad_r, ref_r)) == 1

        # Klasifikasi: prob > 1 harus gagal
        ref_k = d / "ref_k.csv"
        pd.DataFrame({"id_transaksi": ["T1", "T2"]}).to_csv(ref_k, index=False)
        ok_k = d / "ok_k.csv"
        pd.DataFrame({"id_transaksi": ["T1", "T2"], "prob": [0.01, 0.9], "tandai": [0, 1]}).to_csv(ok_k, index=False)
        assert validate_klasifikasi(ok_k, ref_k) == []
        bad_k = d / "bad_k.csv"
        pd.DataFrame({"id_transaksi": ["T1", "T2"], "prob": [1.2, 0.5], "tandai": [0, 1]}).to_csv(bad_k, index=False)
        assert len(validate_klasifikasi(bad_k, ref_k)) == 1

        # Clustering: valid dengan -1 anomali + >=2 klaster
        ref_c = d / "ref_c.csv"
        pd.DataFrame({"id_pelanggan": ["C1", "C2", "C3"]}).to_csv(ref_c, index=False)
        ok_c = d / "ok_c.csv"
        pd.DataFrame({"id_pelanggan": ["C1", "C2", "C3"], "cluster": [-1, 0, 1]}).to_csv(ok_c, index=False)
        assert validate_clustering(ok_c, ref_c) == []


if __name__ == "__main__":
    test_metrics()
    test_validators_catch_contract_violations()
    print("ALL CORE TESTS PASSED")
