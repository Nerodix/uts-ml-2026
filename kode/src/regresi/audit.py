"""Audit Data Tugas 1a — Regresi Estimasi Durasi Pengiriman.

Dibangun bertahap dari nol dengan prinsip satu fungsi untuk satu pertanyaan audit:
- Langkah 1a.1: Cek skema kolom, duplikat/irisan ID, keacakan ID, dan rentang waktu.
- Langkah 1a.2: Cek tipe data, lokasi nilai kosong (missing), dan pola missingness.
- Langkah 1a.3: Screening leakage & pergeseran distribusi di seluruh kolom numerik.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Pastikan direktori kode/ terbaca saat script dijalankan langsung
KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    REGRESI_TEST_PATH,
    REGRESI_TRAIN_PATH,
    SEED,
    ensure_dirs,
)
from src.utils import configure_plotting  # noqa: E402


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


def cek_tipe_data(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Kelompokkan kolom numerik vs teks/kategorikal dan cek konsistensi tipe train–test.

    Pemeriksaan ini memastikan tidak ada kolom angka yang gagal terbaca sebagai
    numerik (misalnya karena terselip string seperti ``'-'`` atau ``'N/A'``) dan
    tipe data antara train dan test selaras.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, Any]
        Daftar kolom teks/kategorikal, jumlah kolom numerik, dan daftar kolom
        dengan tipe data berbeda antara train dan test (jika ada).
    """
    shared_cols = [c for c in test.columns if c in train.columns]
    numeric_cols = [c for c in shared_cols if pd.api.types.is_numeric_dtype(train[c])]
    text_cols = [c for c in shared_cols if c not in numeric_cols]
    dtype_mismatches = {
        c: {"train": str(train[c].dtype), "test": str(test[c].dtype)}
        for c in shared_cols
        if str(train[c].dtype) != str(test[c].dtype)
    }
    return {
        "n_numeric_features": len(numeric_cols),
        "text_columns": text_cols,
        "dtype_mismatches": dtype_mismatches,
        "target_dtype": str(train["durasi_jam"].dtype),
    }


def cek_missing_values(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Identifikasi seluruh kolom yang memiliki nilai kosong (``NaN``) pada train atau test.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, dict[str, float | int]]
        Pemetaan ``nama_kolom -> {train_n, train_pct, test_n, test_pct}`` hanya
        untuk kolom yang memiliki minimal satu nilai kosong.
    """
    hasil: dict[str, dict[str, float | int]] = {}
    for col in train.columns:
        n_tr = int(train[col].isna().sum())
        n_te = int(test[col].isna().sum()) if col in test.columns else 0
        if n_tr > 0 or n_te > 0:
            hasil[col] = {
                "train_n": n_tr,
                "train_pct": round(n_tr / len(train) * 100.0, 2),
                "test_n": n_te,
                "test_pct": round(n_te / len(test) * 100.0, 2) if col in test.columns else 0.0,
            }
    return hasil


def cek_pola_missing(train: pd.DataFrame, kolom: str) -> dict[str, Any]:
    """Selidiki apakah nilai kosong pada ``kolom`` bersifat acak atau berpola (MAR/MNAR).

    Mengevaluasi tiga bukti sekaligus:
    1. Persentase nilai kosong per kategori utama (``layanan``, ``mitra_kurir``,
       ``hub_asal``, ``pulau_tujuan``).
    2. Perbedaan median ``durasi_jam`` saat ``kolom`` kosong vs terisi.
    3. Tiga kolom numerik lain yang paling berkorelasi (Spearman) dengan ``kolom``
       sebagai kandidat fitur bantu imputasi.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    kolom : str
        Nama kolom yang memiliki nilai kosong untuk diselidiki.

    Returns
    -------
    dict[str, Any]
        Rincian % kosong per kategori, median target (kosong vs terisi), serta
        top-3 korelasi Spearman terhadap fitur numerik lain.
    """
    is_miss = train[kolom].isna()

    # 1. Cek % missing per kategori utama (pakai groupby pada masker boolean)
    by_category: dict[str, dict[str, float]] = {}
    for cat_col in ("layanan", "hub_asal", "pulau_tujuan", "mitra_kurir"):
        rates = (is_miss.groupby(train[cat_col]).mean() * 100.0).round(2).to_dict()
        by_category[cat_col] = rates

    # 2. Bandingkan median target saat kosong vs terisi (tahan outlier)
    med_when_missing = round(float(train.loc[is_miss, "durasi_jam"].median()), 2)
    med_when_present = round(float(train.loc[~is_miss, "durasi_jam"].median()), 2)

    # 3. Cari fitur numerik lain yang paling berkorelasi dengan kolom ini
    num_candidates = [
        c for c in train.columns
        if c not in (kolom, "durasi_jam") and pd.api.types.is_numeric_dtype(train[c])
    ]
    corrs = {
        c: round(float(train[kolom].corr(train[c], method="spearman")), 4)
        for c in num_candidates
    }
    top3_corr = dict(
        sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    )

    return {
        "missing_pct_by_category": by_category,
        "median_target_when_missing": med_when_missing,
        "median_target_when_present": med_when_present,
        "top3_spearman_correlates": top3_corr,
    }


def audit_tipe_dan_missing(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Jalankan seluruh pemeriksaan Langkah 1a.2 dengan memanggil 3 fungsi atomik.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, Any]
        Ringkasan tipe data, daftar kolom kosong, dan pola missing untuk tiap kolom kosong.
    """
    tipe = cek_tipe_data(train, test)
    missing = cek_missing_values(train, test)
    pola = {col: cek_pola_missing(train, col) for col in missing}
    return {
        "tipe_data": tipe,
        "missing_values": missing,
        "pola_missing": pola,
    }


def screening_kolom_numerik(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Pindai seluruh kolom numerik dengan dua sinyal: KS (train vs test) dan Spearman ke target.

    Alat detektif objektif untuk menemukan fitur bocor (leakage), fitur yang
    bergeser seiring waktu (drift), dan fitur noise tanpa menebak-nebak.
    Korelasi menggunakan Spearman agar tahan terhadap outlier target yang belum
    dibersihkan pada tahap ini.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi yang memuat kolom target ``durasi_jam``.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    pd.DataFrame
        Tabel satu baris per kolom numerik berisi ``ks_train_vs_test``,
        ``spearman_target``, serta ``min``/``median``/``max`` train dan test.
    """
    num_cols = [
        c for c in test.columns
        if c in train.columns and pd.api.types.is_numeric_dtype(train[c])
    ]
    # Sampling deterministik agar uji KS cepat sekaligus stabil di semua kolom
    tr_samp = train.sample(min(40_000, len(train)), random_state=SEED)
    te_samp = test.sample(min(40_000, len(test)), random_state=SEED)

    rows: list[dict[str, Any]] = []
    for col in num_cols:
        tr_vals = train[col].dropna()
        te_vals = test[col].dropna()
        ks_stat = float(
            stats.ks_2samp(tr_samp[col].dropna(), te_samp[col].dropna()).statistic
        )
        rho = float(train[col].corr(train["durasi_jam"], method="spearman"))
        rows.append(
            {
                "kolom": col,
                "ks_train_vs_test": round(ks_stat, 4),
                "spearman_target": round(rho, 4),
                "train_min": round(float(tr_vals.min()), 3),
                "train_median": round(float(tr_vals.median()), 3),
                "train_max": round(float(tr_vals.max()), 3),
                "test_min": round(float(te_vals.min()), 3),
                "test_median": round(float(te_vals.median()), 3),
                "test_max": round(float(te_vals.max()), 3),
            }
        )
    return pd.DataFrame(rows)


def kelompokkan_kuadran_detektif(tabel_screening: pd.DataFrame) -> dict[str, list[str]]:
    """Kelompokkan hasil screening ke kategori leakage, drift, sinyal sehat, dan noise.

    Aturan ambang eksplisit:
    - ``tersangka_leakage``: ``ks >= 0.50`` dan ``|spearman| >= 0.30``
    - ``drift_bersinyal``  : ``0.05 <= ks < 0.50`` dan ``|spearman| >= 0.02``
    - ``drift_noise``      : ``ks >= 0.05`` dan ``|spearman| < 0.02``
    - ``sinyal_stabil``    : ``ks < 0.05`` dan ``|spearman| >= 0.02``
    - ``noise_stabil``     : ``ks < 0.05`` dan ``|spearman| < 0.02``

    Parameters
    ----------
    tabel_screening : pd.DataFrame
        Tabel keluaran dari :func:`screening_kolom_numerik`.

    Returns
    -------
    dict[str, list[str]]
        Daftar nama kolom pada masing-masing kategori (diurutkan dari KS/Spearman terbesar).
    """
    df = tabel_screening.copy()
    df["abs_rho"] = df["spearman_target"].abs()

    leak = df[(df["ks_train_vs_test"] >= 0.50) & (df["abs_rho"] >= 0.30)].sort_values(
        "ks_train_vs_test", ascending=False
    )
    drift_sig = df[
        (df["ks_train_vs_test"] >= 0.05) & (df["ks_train_vs_test"] < 0.50) & (df["abs_rho"] >= 0.02)
    ].sort_values("ks_train_vs_test", ascending=False)
    drift_noise = df[(df["ks_train_vs_test"] >= 0.05) & (df["abs_rho"] < 0.02)].sort_values(
        "ks_train_vs_test", ascending=False
    )
    sig_stabil = df[(df["ks_train_vs_test"] < 0.05) & (df["abs_rho"] >= 0.02)].sort_values(
        "abs_rho", ascending=False
    )
    noise_stabil = df[(df["ks_train_vs_test"] < 0.05) & (df["abs_rho"] < 0.02)].sort_values(
        "kolom"
    )

    return {
        "tersangka_leakage": leak["kolom"].tolist(),
        "drift_bersinyal": drift_sig["kolom"].tolist(),
        "drift_noise": drift_noise["kolom"].tolist(),
        "sinyal_stabil": sig_stabil["kolom"].tolist(),
        "noise_stabil": noise_stabil["kolom"].tolist(),
    }


def plot_bukti_leakage_scan(train: pd.DataFrame, test: pd.DataFrame) -> Path:
    """Buat grafik 2-panel bukti leakage ``jumlah_scan`` (distribusi train–test & scatter).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    Path
        Path file PNG yang tersimpan di ``kode/figures/task1_1a3_leakage_scan.png``.
    """
    ensure_dirs()
    configure_plotting()

    ks_scan = float(stats.ks_2samp(train["jumlah_scan"], test["jumlah_scan"]).statistic)
    rho_scan = float(train["jumlah_scan"].corr(train["durasi_jam"], method="spearman"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Panel (a): Distribusi jumlah_scan train vs test
    bins = np.arange(0.5, 36.5, 1.0)
    axes[0].hist(
        np.clip(train["jumlah_scan"], 1, 35),
        bins=bins,
        density=True,
        alpha=0.65,
        color="#2563eb",
        label=f"Train (med={train['jumlah_scan'].median():.0f}, max={train['jumlah_scan'].max()})",
    )
    axes[0].hist(
        np.clip(test["jumlah_scan"], 1, 35),
        bins=bins,
        density=True,
        alpha=0.75,
        color="#dc2626",
        label=f"Test (med={test['jumlah_scan'].median():.0f}, max={test['jumlah_scan'].max()})",
    )
    axes[0].set_title(f"(a) Sebaran jumlah_scan Train vs Test (KS = {ks_scan:.3f})")
    axes[0].set_xlabel("jumlah_scan (dipotong tampilan di 35)")
    axes[0].set_ylabel("Densitas")
    axes[0].legend(fontsize=8.5)

    # Panel (b): Scatter jumlah_scan vs durasi_jam (sampel 5.000 baris, sumbu-y dibatasi 350 jam)
    samp = train.sample(5_000, random_state=SEED)
    axes[1].scatter(
        samp["jumlah_scan"],
        samp["durasi_jam"],
        s=7,
        alpha=0.22,
        color="#1d4ed8",
        edgecolors="none",
    )
    axes[1].set_title(f"(b) jumlah_scan vs durasi_jam (Spearman = {rho_scan:.3f})")
    axes[1].set_xlabel("jumlah_scan pada data latih")
    axes[1].set_ylabel("durasi_jam (jam, tampilan 0–350)")
    axes[1].set_ylim(0, 350)

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1a3_leakage_scan.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


DAFTAR_CEK = ("skema", "duplikat", "id", "waktu", "tipe", "missing", "leakage")


def _tampilkan_skema(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil ``cek_skema_kolom``."""
    skema = cek_skema_kolom(train, test)
    print("=== [skema] UKURAN & DAFTAR KOLOM ===")
    print(f"Ukuran train         : {skema['train_shape'][0]:,} baris x {skema['train_shape'][1]} kolom")
    print(f"Ukuran test          : {skema['test_shape'][0]:,} baris x {skema['test_shape'][1]} kolom")
    print(f"Kolom hanya di train : {skema['train_only_cols']}")
    print(f"Kolom hanya di test  : {skema['test_only_cols']}")


def _tampilkan_duplikat(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil ``cek_duplikat_dan_irisan_id``."""
    dup = cek_duplikat_dan_irisan_id(train, test)
    print("=== [duplikat] DUPLIKAT & IRISAN ID ===")
    print(f"Duplikat ID (train / test) : {dup['dup_id_train']} / {dup['dup_id_test']}")
    print(f"Irisan ID train & test     : {dup['overlap_id_train_test']}")
    print(f"Duplikat baris penuh train : {dup['dup_rows_train']}")


def _tampilkan_keacakan_id(train: pd.DataFrame, _test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil ``cek_keacakan_id``."""
    acak = cek_keacakan_id(train)
    print("=== [id] KEACAKAN NOMOR ID ===")
    print(f"Korelasi ID vs urutan baris: {acak['id_corr_row_index']:+.5f}")
    print(f"Korelasi ID vs target      : {acak['id_corr_target_spearman']:+.5f} (Spearman)")


def _tampilkan_waktu(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil ``cek_rentang_waktu``."""
    waktu = cek_rentang_waktu(train, test)
    print("=== [waktu] RENTANG TANGGAL & VOLUME BULANAN ===")
    print(f"Tanggal train : {waktu['train_date_range'][0]} s.d. {waktu['train_date_range'][1]} (rusak={waktu['invalid_date_train']})")
    print(f"Tanggal test  : {waktu['test_date_range'][0]} s.d. {waktu['test_date_range'][1]} (rusak={waktu['invalid_date_test']})")
    print("Volume paket per bulan (train):")
    for bulan, jumlah in waktu["train_monthly_counts"].items():
        print(f"  {bulan} : {jumlah:>6,}")
    print("Volume paket per bulan (test):")
    for bulan, jumlah in waktu["test_monthly_counts"].items():
        print(f"  {bulan} : {jumlah:>6,}")


def _tampilkan_tipe(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil ``cek_tipe_data``."""
    tipe = cek_tipe_data(train, test)
    print("=== [tipe] TIPE DATA KOLOM ===")
    print(f"Jumlah fitur numerik     : {tipe['n_numeric_features']}")
    print(f"Kolom teks / kategorikal : {tipe['text_columns']}")
    print(f"Tipe target durasi_jam   : {tipe['target_dtype']}")
    print(f"Beda tipe train vs test  : {tipe['dtype_mismatches']}")


def _tampilkan_missing(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil ``cek_missing_values`` beserta ``cek_pola_missing``."""
    missing = cek_missing_values(train, test)
    print("=== [missing] NILAI KOSONG & POLA MISSINGNESS ===")
    print(f"Kolom dengan nilai kosong: {list(missing.keys())}")
    for col, stat in missing.items():
        pola = cek_pola_missing(train, col)
        print(f"\n  [Detail missing: {col}]")
        print(f"    Train kosong : {stat['train_n']:,} baris ({stat['train_pct']}%)")
        print(f"    Test kosong  : {stat['test_n']:,} baris ({stat['test_pct']}%)")
        print(f"    % kosong per layanan     : {pola['missing_pct_by_category']['layanan']}")
        print(f"    % kosong per hub_asal    : {pola['missing_pct_by_category']['hub_asal']}")
        print(f"    % kosong per pulau_tujuan: {pola['missing_pct_by_category']['pulau_tujuan']}")
        print(
            f"    Median durasi_jam (kosong vs terisi): "
            f"{pola['median_target_when_missing']} jam vs {pola['median_target_when_present']} jam"
        )
        print(f"    Top-3 fitur paling berkorelasi (Spearman): {pola['top3_spearman_correlates']}")


def _tampilkan_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil Langkah 1a.3 (screening KS & Spearman + figur bukti)."""
    tabel = screening_kolom_numerik(train, test)
    kuadran = kelompokkan_kuadran_detektif(tabel)
    fig_path = plot_bukti_leakage_scan(train, test)

    print("=== [leakage] SCREENING 68 KOLOM NUMERIK (KS TRAIN-VS-TEST & SPEARMAN) ===")
    print("\nTop 10 kolom paling beda antara Train vs Test (KS terbesar):")
    top_ks = tabel.sort_values("ks_train_vs_test", ascending=False).head(10)
    print(top_ks.to_string(index=False))

    print("\nTop 10 kolom paling kuat hubungannya dengan durasi_jam (|Spearman| terbesar):")
    top_rho = tabel.reindex(
        tabel["spearman_target"].abs().sort_values(ascending=False).index
    ).head(10)
    print(top_rho[["kolom", "spearman_target", "ks_train_vs_test", "train_median", "test_median"]].to_string(index=False))

    print("\nPengelompokan otomatis ke kuadran detektif:")
    print(f"  1. Tersangka LEAKAGE (KS>=0.50 & |rho|>=0.30) : {kuadran['tersangka_leakage']}")
    print(f"  2. Drift bersinyal   (KS>=0.05 & |rho|>=0.02) : {kuadran['drift_bersinyal']}")
    print(f"  3. Drift NOISE       (KS>=0.05 & |rho|<0.02)  : {kuadran['drift_noise']}")
    print(f"  4. Sinyal stabil     (KS<0.05  & |rho|>=0.02) : {len(kuadran['sinyal_stabil'])} kolom -> {kuadran['sinyal_stabil']}")
    print(f"  5. Noise stabil      (KS<0.05  & |rho|<0.02)  : {len(kuadran['noise_stabil'])} kolom")
    print(f"\nGrafik bukti tersimpan di: {fig_path.relative_to(KODE_DIR.parent)}")


def main() -> None:
    """Jalankan satu, beberapa, atau seluruh pemeriksaan audit Tugas 1a lewat CLI.

    Contoh penggunaan:
        python kode/src/regresi/audit.py                  # Jalankan semua pemeriksaan
        python kode/src/regresi/audit.py --cek missing    # Hanya cek nilai kosong
        python kode/src/regresi/audit.py --cek leakage    # Hanya screening leakage & drift
    """
    parser = argparse.ArgumentParser(
        description="Audit data Tugas 1a (Regresi). Gunakan --cek untuk memilih fungsi tertentu."
    )
    parser.add_argument(
        "--cek",
        nargs="+",
        choices=DAFTAR_CEK,
        default=list(DAFTAR_CEK),
        help=f"Pilih satu atau lebih pemeriksaan (default: semua -> {', '.join(DAFTAR_CEK)}).",
    )
    args = parser.parse_args()

    train = pd.read_csv(REGRESI_TRAIN_PATH)
    test = pd.read_csv(REGRESI_TEST_PATH)

    handlers = {
        "skema": _tampilkan_skema,
        "duplikat": _tampilkan_duplikat,
        "id": _tampilkan_keacakan_id,
        "waktu": _tampilkan_waktu,
        "tipe": _tampilkan_tipe,
        "missing": _tampilkan_missing,
        "leakage": _tampilkan_leakage,
    }

    for i, nama in enumerate(args.cek):
        if i > 0:
            print()
        handlers[nama](train, test)


if __name__ == "__main__":
    main()
