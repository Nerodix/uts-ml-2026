"""Validator otomatis untuk memastikan ketiga file prediksi sesuai kontrak UTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_KODE_DIR = Path(__file__).resolve().parents[1]
if str(_KODE_DIR) not in sys.path:
    sys.path.insert(0, str(_KODE_DIR))

from src.config import (  # noqa: E402
    CLUSTERING_PATH,
    KLASIFIKASI_TEST_PATH,
    PRED_CLUSTER_PATH,
    PRED_KLASIFIKASI_PATH,
    PRED_REGRESI_PATH,
    REGRESI_TEST_PATH,
)


def _check_ids(pred_ids: pd.Series, ref_ids: pd.Series, name: str, id_col: str) -> list[str]:
    errors: list[str] = []
    if len(pred_ids) != len(ref_ids):
        errors.append(f"[{name}] Jumlah baris {len(pred_ids):,} != ekspektasi {len(ref_ids):,}")
    if pred_ids.isna().any():
        errors.append(f"[{name}] Kolom {id_col} mengandung NaN")
    if pred_ids.duplicated().any():
        errors.append(f"[{name}] Kolom {id_col} mengandung duplikat")
    if set(pred_ids) != set(ref_ids):
        missing = len(set(ref_ids) - set(pred_ids))
        extra = len(set(pred_ids) - set(ref_ids))
        errors.append(f"[{name}] Himpunan {id_col} tidak cocok (missing={missing}, extra={extra})")
    elif not pred_ids.reset_index(drop=True).equals(ref_ids.reset_index(drop=True)):
        errors.append(f"[{name}] Urutan {id_col} berbeda dari file data uji (disarankan identik)")
    return errors


def validate_regresi(
    pred_path: Path = PRED_REGRESI_PATH,
    test_path: Path = REGRESI_TEST_PATH,
) -> list[str]:
    name = "prediksi_regresi.csv"
    if not pred_path.exists():
        return [f"[{name}] File tidak ditemukan di {pred_path}"]

    df = pd.read_csv(pred_path)
    expected_cols = ["id_paket", "p50", "p90"]
    if list(df.columns) != expected_cols:
        return [f"[{name}] Kolom {list(df.columns)} != {expected_cols}"]

    ref = pd.read_csv(test_path, usecols=["id_paket"])
    errors = _check_ids(df["id_paket"], ref["id_paket"], name, "id_paket")

    for col in ("p50", "p90"):
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals).all():
            errors.append(f"[{name}] Kolom {col} mengandung NaN atau non-finite")
        elif (vals <= 0).any():
            errors.append(f"[{name}] Kolom {col} mengandung nilai <= 0 (min={vals.min():.4f})")

    if not errors and (df["p90"] < df["p50"]).any():
        n_bad = int((df["p90"] < df["p50"]).sum())
        errors.append(f"[{name}] Terdapat {n_bad:,} baris dengan p90 < p50")
    return errors


def validate_klasifikasi(
    pred_path: Path = PRED_KLASIFIKASI_PATH,
    test_path: Path = KLASIFIKASI_TEST_PATH,
) -> list[str]:
    name = "prediksi_klasifikasi.csv"
    if not pred_path.exists():
        return [f"[{name}] File tidak ditemukan di {pred_path}"]

    df = pd.read_csv(pred_path)
    expected_cols = ["id_transaksi", "prob", "tandai"]
    if list(df.columns) != expected_cols:
        return [f"[{name}] Kolom {list(df.columns)} != {expected_cols}"]

    ref = pd.read_csv(test_path, usecols=["id_transaksi"])
    errors = _check_ids(df["id_transaksi"], ref["id_transaksi"], name, "id_transaksi")

    prob = pd.to_numeric(df["prob"], errors="coerce")
    if prob.isna().any() or not np.isfinite(prob).all():
        errors.append(f"[{name}] Kolom prob mengandung NaN atau non-finite")
    elif (prob < 0.0).any() or (prob > 1.0).any():
        errors.append(f"[{name}] Kolom prob di luar [0, 1] (min={prob.min()}, max={prob.max()})")

    tandai = df["tandai"]
    if tandai.isna().any() or not set(tandai.unique()).issubset({0, 1}):
        errors.append(f"[{name}] Kolom tandai harus 0/1 integer (nilai unik: {tandai.unique()[:10]})")
    elif not pd.api.types.is_integer_dtype(tandai):
        errors.append(f"[{name}] Kolom tandai bukan bertipe integer (dtype={tandai.dtype})")
    return errors


def validate_clustering(
    pred_path: Path = PRED_CLUSTER_PATH,
    data_path: Path = CLUSTERING_PATH,
) -> list[str]:
    name = "cluster.csv"
    if not pred_path.exists():
        return [f"[{name}] File tidak ditemukan di {pred_path}"]

    df = pd.read_csv(pred_path)
    expected_cols = ["id_pelanggan", "cluster"]
    if list(df.columns) != expected_cols:
        return [f"[{name}] Kolom {list(df.columns)} != {expected_cols}"]

    ref = pd.read_csv(data_path, usecols=["id_pelanggan"])
    errors = _check_ids(df["id_pelanggan"], ref["id_pelanggan"], name, "id_pelanggan")

    cl = df["cluster"]
    if cl.isna().any():
        errors.append(f"[{name}] Kolom cluster mengandung NaN")
    elif not pd.api.types.is_integer_dtype(cl):
        errors.append(f"[{name}] Kolom cluster bukan integer (dtype={cl.dtype})")
    else:
        if (cl < -1).any():
            errors.append(f"[{name}] Kolom cluster mengandung nilai < -1 (min={cl.min()})")
        non_anom = cl[cl != -1].nunique()
        if non_anom < 2:
            errors.append(f"[{name}] Jumlah klaster non-anomali terlalu sedikit ({non_anom})")
    return errors


def validate_all(tasks: list[str] | None = None) -> int:
    selected = set(tasks or ["regresi", "klasifikasi", "clustering"])
    all_errors: list[str] = []

    if "regresi" in selected:
        all_errors.extend(validate_regresi())
    if "klasifikasi" in selected:
        all_errors.extend(validate_klasifikasi())
    if "clustering" in selected:
        all_errors.extend(validate_clustering())

    if all_errors:
        print("VALIDASI GAGAL:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"VALIDASI LULUS untuk: {', '.join(sorted(selected))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi format file prediksi UTS ML 2026.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["regresi", "klasifikasi", "clustering"],
        default=["regresi", "klasifikasi", "clustering"],
        help="Daftar task yang ingin divalidasi.",
    )
    args = parser.parse_args()
    return validate_all(args.tasks)


if __name__ == "__main__":
    sys.exit(main())
