"""Pengujian unit skema validasi waktu anti-censoring Tugas 2b."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.klasifikasi.preprocess import MATURE_TRAIN_END_DATE, SNAPSHOT_TIMESTAMP  # noqa: E402
from src.klasifikasi.validation import (  # noqa: E402
    MIN_MATURITY_LAG_DAYS,
    buat_fold_waktu_matang,
    buat_split_shift_agustus,
    filter_label_matang,
)


def _buat_df_sintetis() -> pd.DataFrame:
    """Buat deret transaksi harian Jan–Agt 2025 dengan urutan baris sengaja diacak."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01 12:00:00", "2025-08-31 12:00:00", freq="6h")
    perm = rng.permutation(len(dates))
    return pd.DataFrame({"waktu": dates[perm].strftime("%Y-%m-%d %H:%M:%S"), "waktu_dt": dates[perm]})


def test_invarian_fold_waktu_matang() -> None:
    """Pastikan seluruh fold ketat kronologis, saling lepas, dan 100% bebas sensor Agustus."""
    df = _buat_df_sintetis()
    folds = buat_fold_waktu_matang(df)
    assert len(folds) == 3

    seen_val: set[int] = set()
    mature_cutoff = pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")
    for f in folds:
        tr_t = df.iloc[f.train_idx]["waktu_dt"]
        va_t = df.iloc[f.val_idx]["waktu_dt"]
        # Tidak ada irisan latih vs validasi
        assert set(f.train_idx).isdisjoint(set(f.val_idx))
        # Latih ketat sebelum validasi
        assert tr_t.max() < va_t.min()
        # Seluruh validasi berada pada zona label matang (<= 31 Jul 2025)
        assert va_t.max() <= mature_cutoff
        # Jendela validasi antar fold tidak tumpang tindih
        assert seen_val.isdisjoint(set(f.val_idx))
        seen_val.update(f.val_idx.tolist())


def test_filter_label_matang_dan_split_shift_agustus() -> None:
    """Pastikan masker maturitas membuang 25 hari terakhir & split Agustus terurut ketat."""
    df = _buat_df_sintetis()
    mask = filter_label_matang(df, min_lag_hari=MIN_MATURITY_LAG_DAYS, snapshot_ts=SNAPSHOT_TIMESTAMP)
    max_kept = df.loc[mask, "waktu_dt"].max()
    assert (SNAPSHOT_TIMESTAMP - max_kept).total_seconds() / 86_400.0 >= MIN_MATURITY_LAG_DAYS

    splits = buat_split_shift_agustus(df)
    t_jj = df.iloc[splits["jan_jul_idx"]]["waktu_dt"]
    t_ad = df.iloc[splits["aug_adapt_idx"]]["waktu_dt"]
    t_va = df.iloc[splits["aug_val_idx"]]["waktu_dt"]
    t_tl = df.iloc[splits["aug_tail_idx"]]["waktu_dt"]
    assert t_jj.max() < t_ad.min()
    assert t_ad.max() < t_va.min()
    assert t_va.max() < t_tl.min()
    assert len(splits["jan_jul_idx"]) + len(splits["aug_adapt_idx"]) + len(splits["aug_val_idx"]) + len(
        splits["aug_tail_idx"]
    ) == len(df)


def main() -> None:
    test_invarian_fold_waktu_matang()
    test_filter_label_matang_dan_split_shift_agustus()
    print("ALL KLASIFIKASI VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
