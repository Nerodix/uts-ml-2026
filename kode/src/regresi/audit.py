"""Audit Data Tugas 1a — Regresi Estimasi Durasi Pengiriman.

Dibangun bertahap dari nol dengan prinsip satu fungsi untuk satu pertanyaan audit:
- Langkah 1a.1: Cek skema kolom, duplikat/irisan ID, keacakan ID, dan rentang waktu.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Pastikan direktori kode/ terbaca saat script dijalankan langsung
KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import REGRESI_TEST_PATH, REGRESI_TRAIN_PATH  # noqa: E402


def cek_skema_kolom(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Periksa ukuran dataset dan kesesuaian daftar kolom train vs test.

    Memastikan perbedaan kolom antara data latih dan data uji hanya berupa
    kolom target ``durasi_jam``, tanpa kolom ekstra atau hilang yang tidak terduga.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih dari ``regresi_train.csv.gz``.
    test : pd.DataFrame
        Data uji dari ``regresi_test.csv.gz``.

    Returns
    -------
    dict[str, Any]
        Dimensi train/test serta daftar kolom yang hanya muncul di salah satu split.
    """
    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "train_only_cols": sorted(set(train.columns) - set(test.columns)),
        "test_only_cols": sorted(set(test.columns) - set(train.columns)),
    }


def cek_duplikat_dan_irisan_id(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """Periksa duplikat ``id_paket``, duplikat baris penuh, dan irisan ID train–test.

    Duplikat baris dapat memalsukan skor validasi, sedangkan irisan ID antara
    data latih dan data uji menandakan kebocoran sampel langsung.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, int]
        Jumlah duplikat ID di train/test, irisan ID antar split, dan duplikat baris train.
    """
    return {
        "dup_id_train": int(train["id_paket"].duplicated().sum()),
        "dup_id_test": int(test["id_paket"].duplicated().sum()),
        "overlap_id_train_test": int(len(set(train["id_paket"]) & set(test["id_paket"]))),
        "dup_rows_train": int(train.duplicated().sum()),
    }


def cek_keacakan_id(train: pd.DataFrame) -> dict[str, float]:
    """Periksa apakah angka pada ``id_paket`` bersifat acak atau membawa pola bocor.

    ID yang dibuat berurutan secara kronologis terkadang membocorkan waktu
    atau target. Korelasi target menggunakan Spearman agar tahan terhadap
    kemungkinan outlier sebelum tahap pembersihan target.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi yang memuat ``id_paket`` dan ``durasi_jam``.

    Returns
    -------
    dict[str, float]
        Korelasi nomor ID terhadap indeks baris dan terhadap ``durasi_jam`` (Spearman).
    """
    id_num = train["id_paket"].str.slice(1).astype(np.int64)
    row_idx = pd.Series(np.arange(len(train), dtype=np.int64))
    return {
        "id_corr_row_index": round(float(id_num.corr(row_idx)), 5),
        "id_corr_target_spearman": round(
            float(id_num.corr(train["durasi_jam"], method="spearman")), 5
        ),
    }


def cek_rentang_waktu(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Verifikasi validitas ``tgl_pickup``, pemisahan waktu train < test, dan volume bulanan.

    Menggunakan ``errors='coerce'`` saat parsing tanggal agar tanggal berformat
    rusak terdeteksi sebagai ``NaT`` alih-alih diabaikan diam-diam.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, Any]
        Jumlah tanggal invalid, rentang tanggal min–maks, dan jumlah paket per bulan.
    """
    dt_tr = pd.to_datetime(train["tgl_pickup"], errors="coerce")
    dt_te = pd.to_datetime(test["tgl_pickup"], errors="coerce")

    return {
        "invalid_date_train": int(dt_tr.isna().sum()),
        "invalid_date_test": int(dt_te.isna().sum()),
        "train_date_range": (str(dt_tr.min().date()), str(dt_tr.max().date())),
        "test_date_range": (str(dt_te.min().date()), str(dt_te.max().date())),
        "train_monthly_counts": dt_tr.dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
        "test_monthly_counts": dt_te.dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
    }


def audit_skema_dan_waktu(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Jalankan seluruh pemeriksaan Langkah 1a.1 dengan memanggil 4 fungsi atomik.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih dari ``regresi_train.csv.gz``.
    test : pd.DataFrame
        Data uji dari ``regresi_test.csv.gz``.

    Returns
    -------
    dict[str, Any]
        Hasil gabungan skema, duplikat/irisan ID, keacakan ID, dan rentang waktu.
    """
    return {
        "skema": cek_skema_kolom(train, test),
        "duplikat": cek_duplikat_dan_irisan_id(train, test),
        "keacakan_id": cek_keacakan_id(train),
        "waktu": cek_rentang_waktu(train, test),
    }


def main() -> None:
    """Jalankan Langkah 1a.1 dan cetak hasilnya ke terminal."""
    train = pd.read_csv(REGRESI_TRAIN_PATH)
    test = pd.read_csv(REGRESI_TEST_PATH)

    hasil = audit_skema_dan_waktu(train, test)
    skema = hasil["skema"]
    dup = hasil["duplikat"]
    acak = hasil["keacakan_id"]
    waktu = hasil["waktu"]

    print("=== LANGKAH 1a.1: SKEMA, DUPLIKAT, ID, DAN WAKTU ===")
    print(f"Ukuran train               : {skema['train_shape'][0]:,} baris x {skema['train_shape'][1]} kolom")
    print(f"Ukuran test                : {skema['test_shape'][0]:,} baris x {skema['test_shape'][1]} kolom")
    print(f"Kolom hanya di train       : {skema['train_only_cols']}")
    print(f"Kolom hanya di test        : {skema['test_only_cols']}")
    print(f"Duplikat ID (train / test) : {dup['dup_id_train']} / {dup['dup_id_test']}")
    print(f"Irisan ID train & test     : {dup['overlap_id_train_test']}")
    print(f"Duplikat baris penuh train : {dup['dup_rows_train']}")
    print(f"Korelasi ID vs urutan baris: {acak['id_corr_row_index']:+.5f}")
    print(f"Korelasi ID vs target      : {acak['id_corr_target_spearman']:+.5f} (Spearman)")
    print(f"Tanggal train              : {waktu['train_date_range'][0]} s.d. {waktu['train_date_range'][1]} (rusak={waktu['invalid_date_train']})")
    print(f"Tanggal test               : {waktu['test_date_range'][0]} s.d. {waktu['test_date_range'][1]} (rusak={waktu['invalid_date_test']})")

    print("\nVolume paket per bulan (train):")
    for bulan, jumlah in waktu["train_monthly_counts"].items():
        print(f"  {bulan} : {jumlah:>6,}")

    print("\nVolume paket per bulan (test):")
    for bulan, jumlah in waktu["test_monthly_counts"].items():
        print(f"  {bulan} : {jumlah:>6,}")


if __name__ == "__main__":
    main()
