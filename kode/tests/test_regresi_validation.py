"""Unit test untuk memastikan fold validasi waktu Tugas 1b bebas kebocoran masa depan."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.regresi.validation import buat_fold_waktu  # noqa: E402


def test_buat_fold_waktu_strictly_chronological() -> None:
    dates = pd.date_range("2025-01-01", "2025-08-31", freq="D")
    df = pd.DataFrame({"tgl_pickup": dates.strftime("%Y-%m-%d")})

    folds = buat_fold_waktu(df)
    assert len(folds) == 3

    prev_train_len = 0
    for f in folds:
        assert len(f.train_idx) > prev_train_len  # Expanding window
        prev_train_len = len(f.train_idx)
        assert len(f.val_idx) > 0
        # Tidak boleh ada irisan baris antara train dan val
        assert len(set(f.train_idx) & set(f.val_idx)) == 0
        # Seluruh tanggal train harus ketat < tanggal minimum val
        max_tr_date = pd.to_datetime(df.iloc[f.train_idx]["tgl_pickup"]).max()
        min_va_date = pd.to_datetime(df.iloc[f.val_idx]["tgl_pickup"]).min()
        assert max_tr_date < min_va_date


if __name__ == "__main__":
    test_buat_fold_waktu_strictly_chronological()
    print("ALL REGRESI VALIDATION TESTS PASSED")
