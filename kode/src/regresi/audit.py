"""Audit Data Tugas 1a — Regresi Estimasi Durasi Pengiriman.

Dibangun bertahap dari nol dengan prinsip satu fungsi untuk satu pertanyaan audit:
- Langkah 1a.1: Cek skema kolom, duplikat/irisan ID, keacakan ID, dan rentang waktu.
- Langkah 1a.2: Cek tipe data, lokasi nilai kosong (missing), dan pola missingness.
- Langkah 1a.3: Screening leakage & pergeseran distribusi di seluruh kolom numerik.
- Langkah 1a.4: Audit anomali target durasi_jam & nilai tidak wajar/sentinel numerik.
- Langkah 1a.5: Audit kolom kategorikal (unseen/high-cardinality) & pergeseran waktu.
- Langkah 1a.6: Pengukuran dampak kuantitatif tiap temuan & ekspor JSON laporan.
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
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.regresi.audit_dampak import jalankan_semua_dampak  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    configure_plotting,
    label_bulan,
    save_json,
    set_global_seed,
    sumbu_koma,
)


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

    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.5))

    # Panel (a): Sebaran jumlah_scan pada data latih dan data uji
    batas = 35
    bins = np.arange(0.5, batas + 1.5, 1.0)
    for data, nama, warna, alpha in [(train, "Data latih", "#2563eb", 0.65), (test, "Data uji", "#dc2626", 0.75)]:
        kolom = data["jumlah_scan"]
        axes[0].hist(
            np.clip(kolom, 1, batas),
            bins=bins,
            density=True,
            alpha=alpha,
            color=warna,
            label=f"{nama} (median {kolom.median():.0f}; maks. {kolom.max():.0f})",
        )
    axes[0].set_title(f"(a) Sebaran jumlah_scan (KS = {angka_id(ks_scan, 3)})")
    axes[0].set_xlabel(f"jumlah_scan (nilai di atas {batas} digabung)")
    axes[0].set_ylabel("Kepadatan")
    sumbu_koma(axes[0], "y")
    axes[0].legend()

    # Panel (b): Diagram pencar jumlah_scan dan durasi_jam (sampel 5.000 baris, sumbu y dibatasi 350 jam)
    samp = train.sample(5_000, random_state=SEED)
    axes[1].scatter(
        samp["jumlah_scan"],
        samp["durasi_jam"],
        s=3,
        alpha=0.22,
        color="#1d4ed8",
        edgecolors="none",
        rasterized=True,
    )
    axes[1].set_title(f"(b) Kaitan dengan durasi_jam (\u03c1 = {angka_id(rho_scan, 3)})")
    axes[1].set_xlabel("jumlah_scan pada data latih")
    axes[1].set_ylabel("durasi_jam (jam)")
    axes[1].set_ylim(0, 350)

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1a3_leakage_scan.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def cek_anomali_target(train: pd.DataFrame, ambang_menit: float = 1000.0) -> dict[str, Any]:
    """Audit sebaran target ``durasi_jam`` dan uji hipotesis error satuan menit (``* 60``).

    Memeriksa kuantil target, celah nilai antara ``<= ambang_menit`` dan
    ``> ambang_menit``, persentase kelipatan eksak 6 (``0,1 jam * 60``), serta
    kesesuaian median dan rasio ``durasi_jam / jumlah_scan`` setelah dibagi 60.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi yang memuat ``durasi_jam`` dan ``jumlah_scan``.
    ambang_menit : float, default=1000.0
        Ambang batas (jam) pemisah antara durasi sah dan kandidat satuan menit.

    Returns
    -------
    dict[str, Any]
        Ringkasan kuantil, celah distribusi, dan bukti kuantitatif error satuan menit.
    """
    y = train["durasi_jam"]
    corrupt_mask = y > ambang_menit
    clean = y.loc[~corrupt_mask]
    corrupt = y.loc[corrupt_mask]
    corrupt_div60 = corrupt / 60.0

    # Cek apakah ada nilai <= 1000 yang mencurigakan (>300 jam tapi jumlah_scan < 15)
    susp_low_scan = int(((y > 300) & (y <= ambang_menit) & (train["jumlah_scan"] < 15)).sum())

    kuantil = {
        str(q): round(float(val), 2)
        for q, val in y.quantile([0.0, 0.01, 0.5, 0.9, 0.99, 0.999, 1.0]).items()
    }
    return {
        "n_le_zero": int((y <= 0).sum()),
        "kuantil_mentah": kuantil,
        "n_gt_ambang": int(corrupt_mask.sum()),
        "pct_gt_ambang": round(float(corrupt_mask.mean() * 100.0), 4),
        "maks_sah_le_ambang": round(float(clean.max()), 2),
        "min_ekstrem_gt_ambang": round(float(corrupt.min()), 2),
        "pct_kelipatan_6": round(float(np.isclose(corrupt % 6.0, 0.0).mean() * 100.0), 2),
        "suspicious_le_ambang_low_scan": susp_low_scan,
        "mean_mentah": round(float(y.mean()), 2),
        "mean_setelah_koreksi": round(float(np.where(corrupt_mask, y / 60.0, y).mean()), 2),
        "median_bersih": round(float(clean.median()), 2),
        "median_ekstrem_dibagi_60": round(float(corrupt_div60.median()), 2),
        "median_jam_per_scan_bersih": round(
            float((clean / train.loc[~corrupt_mask, "jumlah_scan"]).median()), 3
        ),
        "median_jam_per_scan_koreksi": round(
            float((corrupt_div60 / train.loc[corrupt_mask, "jumlah_scan"]).median()), 3
        ),
    }


def plot_bukti_target_menit(train: pd.DataFrame, ambang_menit: float = 1000.0) -> Path:
    """Buat grafik 2-panel bukti error satuan menit pada target ``durasi_jam``.

    Panel (a) menampilkan histogram ``log10(durasi_jam)`` yang memperlihatkan
    celah kosong bersih di sekitar 1.000 jam. Panel (b) membandingkan sebaran
    baris ``> 1000`` setelah dibagi 60 terhadap data bersih.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    ambang_menit : float, default=1000.0
        Ambang batas pemisah jam vs menit.

    Returns
    -------
    Path
        Path file PNG yang tersimpan di ``kode/figures/task1_1a4_target_unit_error.png``.
    """
    ensure_dirs()
    configure_plotting()

    y = train["durasi_jam"]
    corrupt_mask = y > ambang_menit
    clean = y.loc[~corrupt_mask]
    corrupt_div60 = y.loc[corrupt_mask] / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.5))

    # Panel (a): Celah kosong pada skala log10
    axes[0].hist(
        np.log10(clean),
        bins=45,
        color="#2563eb",
        alpha=0.75,
        label=f"Wajar (paling lama {angka_id(clean.max(), 1)} jam)",
    )
    axes[0].hist(
        np.log10(y.loc[corrupt_mask]),
        bins=20,
        color="#dc2626",
        alpha=0.85,
        label=f"Tercatat dalam menit (n = {angka_id(int(corrupt_mask.sum()), 0)})",
    )
    axes[0].axvline(
        np.log10(ambang_menit),
        color="#111827",
        linestyle="--",
        linewidth=0.9,
        label=f"Ambang {angka_id(ambang_menit, 0)} jam",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Celah pada durasi_jam Mentah")
    axes[0].set_xlabel(r"$\log_{10}$(durasi_jam mentah)")
    axes[0].set_ylabel("Jumlah paket (skala log)")
    sumbu_koma(axes[0], "x")
    axes[0].legend()

    # Panel (b): Sebaran sesudah nilai di atas ambang dibagi 60
    axes[1].hist(
        clean,
        bins=60,
        range=(0, 260),
        density=True,
        alpha=0.55,
        color="#2563eb",
        label=f"Data wajar (median {angka_id(clean.median(), 1)} jam)",
    )
    axes[1].hist(
        corrupt_div60,
        bins=30,
        range=(0, 260),
        density=True,
        alpha=0.65,
        color="#16a34a",
        label=f"Di atas ambang, dibagi 60 (median {angka_id(corrupt_div60.median(), 1)} jam)",
    )
    axes[1].set_title("(b) Sebaran setelah Dibagi 60")
    axes[1].set_xlabel("durasi_jam terkoreksi (jam)")
    axes[1].set_ylabel("Kepadatan")
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.22)
    sumbu_koma(axes[1], "y")
    axes[1].legend()

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1a4_target_unit_error.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def cek_batas_fisik_numerik(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Periksa rentang min/median/max dan pelanggaran batas fisik fitur numerik bernama.

    Memeriksa seluruh fitur numerik bernama (bukan kolom anonim ``f01–f50`` dan
    bukan ``jumlah_scan`` yang sudah teridentifikasi bocor) terhadap aturan fisik
    seperti nilai negatif, skor di luar ``[0, 1]``, rating di luar ``[1, 5]``,
    jam di luar ``[0, 23]``, dan sentinel jarak ``0`` atau ``9999``.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    pd.DataFrame
        Tabel ringkasan batas nilai dan catatan anomali per fitur numerik bernama.
    """
    named_nums = [
        c for c in test.columns
        if pd.api.types.is_numeric_dtype(test[c])
        and not (c.startswith("f") and c[1:].isdigit())
        and c != "jumlah_scan"
    ]

    aturan: dict[str, tuple[str, Any]] = {
        "jam_pickup": ("di luar [0,23]", lambda s: (s < 0) | (s > 23)),
        "jarak_km": ("==0 atau >=9999", lambda s: (s <= 0) | (s >= 9999)),
        "berat_kg": ("<=0", lambda s: s <= 0),
        "volume_liter": ("<=0", lambda s: s <= 0),
        "jumlah_item": ("<1", lambda s: s < 1),
        "nilai_barang": ("<=0", lambda s: s <= 0),
        "asuransi": ("bukan {0,1}", lambda s: ~s.isin([0, 1])),
        "cod": ("bukan {0,1}", lambda s: ~s.isin([0, 1])),
        "alamat_skor": ("di luar [0,1]", lambda s: (s < 0) | (s > 1)),
        "indeks_kemacetan": ("di luar [0,1]", lambda s: (s < 0) | (s > 1)),
        "rating_penjual": ("di luar [1,5]", lambda s: (s < 1) | (s > 5)),
    }

    rows: list[dict[str, Any]] = []
    for col in named_nums:
        tr_v = train[col].dropna()
        te_v = test[col].dropna()
        label_aturan, fn = aturan.get(col, ("<0", lambda s: s < 0))
        n_bad_tr = int(fn(tr_v).sum())
        n_bad_te = int(fn(te_v).sum())
        rows.append(
            {
                "kolom": col,
                "train_min": round(float(tr_v.min()), 3),
                "train_med": round(float(tr_v.median()), 3),
                "train_max": round(float(tr_v.max()), 3),
                "test_min": round(float(te_v.min()), 3),
                "test_max": round(float(te_v.max()), 3),
                "cek_fisik": label_aturan,
                "melanggar_tr": n_bad_tr,
                "melanggar_te": n_bad_te,
            }
        )
    return pd.DataFrame(rows)


def cek_pemulihan_rute_jarak(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Audit sentinel ``jarak_km`` (0 & 9999) dan uji pemulihan eksak via rute train.

    Memeriksa seberapa konsisten ``jarak_km`` pada tiap pasangan
    ``(kota_asal, kota_tujuan)`` yang valid di data latih, serta berapa banyak
    baris sentinel pada data uji yang dapat dipulihkan persis tanpa mengintip test.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, Any]
        Jumlah sentinel train/test, bukti determinisme rute (``max_std``), dan
        persentase cakupan pemulihan rute pada data uji.
    """
    bad_tr = (train["jarak_km"] <= 0) | (train["jarak_km"] >= 9999)
    bad_te = (test["jarak_km"] <= 0) | (test["jarak_km"] >= 9999)

    valid_tr = train.loc[~bad_tr]
    route_stats = valid_tr.groupby(["kota_asal", "kota_tujuan"])["jarak_km"].agg(["nunique", "std"])
    valid_routes = set(zip(valid_tr["kota_asal"], valid_tr["kota_tujuan"]))

    test_bad_df = test.loc[bad_te]
    recoverable = sum(
        (a, b) in valid_routes
        for a, b in zip(test_bad_df["kota_asal"], test_bad_df["kota_tujuan"])
    )
    total_bad_te = int(bad_te.sum())

    return {
        "train_nol": int((train["jarak_km"] == 0).sum()),
        "train_9999": int((train["jarak_km"] == 9999).sum()),
        "train_total_rusak": int(bad_tr.sum()),
        "train_pct_rusak": round(float(bad_tr.mean() * 100.0), 3),
        "test_nol": int((test["jarak_km"] == 0).sum()),
        "test_9999": int((test["jarak_km"] == 9999).sum()),
        "test_total_rusak": total_bad_te,
        "test_pct_rusak": round(float(bad_te.mean() * 100.0), 3),
        "jumlah_rute_valid_train": int(len(route_stats)),
        "max_nunique_per_rute": int(route_stats["nunique"].max()),
        "max_std_per_rute": round(float(route_stats["std"].fillna(0.0).max()), 6),
        "test_rusak_terpulihkan_eksak": int(recoverable),
        "test_rusak_terpulihkan_pct": round(recoverable / max(1, total_bad_te) * 100.0, 2),
    }


KOLOM_KATEGORIKAL = (
    "layanan",
    "mitra_kurir",
    "hub_asal",
    "kota_asal",
    "pulau_tujuan",
    "kota_tujuan",
)


def cek_kolom_kategorikal(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Audit 6 kolom kategorikal: kardinalitas, typo/spasi, kategori unseen di test, dan hierarki.

    Memeriksa apakah ada inkonsistensi kapitalisasi/spasi (membandingkan jumlah
    unik mentah vs sesudah ``strip().lower()``), mendata kategori yang hanya muncul
    pada data uji beserta porsi barisnya, serta memverifikasi hubungan 1-ke-1
    ``kota_asal -> hub_asal`` dan ``kota_tujuan -> pulau_tujuan``.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    dict[str, Any]
        Tabel ringkasan per kolom kategorikal dan cek konsistensi hierarki kota.
    """
    rows: list[dict[str, Any]] = []
    for col in KOLOM_KATEGORIKAL:
        tr_s = train[col].dropna().astype(str)
        te_s = test[col].dropna().astype(str)
        tr_u = set(tr_s.unique())
        te_u = set(te_s.unique())
        unseen = sorted(te_u - tr_u)
        unseen_rows = int(test[col].astype(str).isin(unseen).sum())
        norm_tr = int(tr_s.str.strip().str.lower().nunique())
        norm_te = int(te_s.str.strip().str.lower().nunique())
        rows.append(
            {
                "kolom": col,
                "unik_train": len(tr_u),
                "unik_train_norm": norm_tr,
                "unik_test": len(te_u),
                "unik_test_norm": norm_te,
                "unseen_di_test": unseen,
                "baris_unseen": unseen_rows,
                "pct_unseen_test": round(unseen_rows / len(test) * 100.0, 2),
            }
        )

    kt_counts = train["kota_tujuan"].value_counts()
    return {
        "ringkasan_kategori": pd.DataFrame(rows),
        "max_hub_per_kota_asal": int(train.groupby("kota_asal")["hub_asal"].nunique().max()),
        "max_pulau_per_kota_tujuan": int(train.groupby("kota_tujuan")["pulau_tujuan"].nunique().max()),
        "min_baris_kota_tujuan_train": int(kt_counts.min()),
        "kota_tujuan_kurang_50_baris": int((kt_counts < 50).sum()),
    }


def cek_pergeseran_bulanan(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Lacak pergeseran bulanan (Jan–Okt 2025) pada kurir, antrean hub, dan fitur drift.

    Menghubungkan temuan fitur drift dari Langkah 1a.3 (``f35``, ``antrean_hub_asal``,
    ``f17``, ``f30``) dengan dimensi waktu bulan-ke-bulan serta membandingkan
    perubahan ``f35`` kurir ``K07`` terhadap median ``durasi_jam`` terkoreksi.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi (Jan–Agu 2025).
    test : pd.DataFrame
        Data uji regresi (Sep–Okt 2025).

    Returns
    -------
    dict[str, Any]
        Pivot bulanan ``f35`` per kurir, ``durasi_jam`` bersih per kurir,
        ``antrean_hub_asal`` per hub, dan median bulanan ``f30`` vs ``f17``.
    """
    tr = train.copy()
    te = test.copy()
    tr["_y"] = np.where(tr["durasi_jam"] > 1000.0, tr["durasi_jam"] / 60.0, tr["durasi_jam"])
    tr["_ym"] = pd.to_datetime(tr["tgl_pickup"]).dt.to_period("M").astype(str)
    te["_ym"] = pd.to_datetime(te["tgl_pickup"]).dt.to_period("M").astype(str)

    both = pd.concat(
        [
            tr[["_ym", "mitra_kurir", "hub_asal", "antrean_hub_asal", "f35", "f30", "f17"]],
            te[["_ym", "mitra_kurir", "hub_asal", "antrean_hub_asal", "f35", "f30", "f17"]],
        ],
        ignore_index=True,
    )

    f35_kurir = both.pivot_table(index="mitra_kurir", columns="_ym", values="f35", aggfunc="median").round(3)
    y_kurir = tr.pivot_table(index="mitra_kurir", columns="_ym", values="_y", aggfunc="median").round(1)
    antrean_hub = both.pivot_table(index="hub_asal", columns="_ym", values="antrean_hub_asal", aggfunc="median").round(0)
    drift_f = both.groupby("_ym")[["antrean_hub_asal", "f17", "f30"]].median().round(3)

    return {
        "f35_per_kurir_bulan": f35_kurir,
        "durasi_per_kurir_bulan": y_kurir,
        "antrean_per_hub_bulan": antrean_hub,
        "ringkasan_drift_bulanan": drift_f,
    }


def cek_efek_kalender(train: pd.DataFrame) -> dict[str, Any]:
    """Ukur efek hari dalam seminggu, tanggal kembar, dan musim libur Lebaran 2025.

    Menggunakan ``durasi_jam`` yang telah dikoreksi satuan menit (``/ 60`` untuk
    ``> 1000``) agar median dan P90 tidak terdistorsi.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi (Jan–Agu 2025).

    Returns
    -------
    dict[str, Any]
        Tabel median & P90 menurut hari dalam seminggu, tanggal kembar (1.1–8.8),
        serta jendela libur Lebaran (26 Mar–6 Apr 2025).
    """
    tr = train.copy()
    tr["_y"] = np.where(tr["durasi_jam"] > 1000.0, tr["durasi_jam"] / 60.0, tr["durasi_jam"])
    dt = pd.to_datetime(tr["tgl_pickup"])
    nama_hari = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
    tr["_hari"] = dt.dt.day_of_week.map(lambda d: f"{d}-{nama_hari[d]}")

    by_hari = (
        tr.groupby("_hari")["_y"]
        .agg(jumlah="size", median="median", p90=lambda s: s.quantile(0.9))
        .round(2)
    )

    is_kembar = dt.dt.month == dt.dt.day
    is_lebaran = (dt >= "2025-03-26") & (dt <= "2025-04-06")

    return {
        "per_hari": by_hari,
        "tanggal_kembar": {
            "n_kembar": int(is_kembar.sum()),
            "median_kembar": round(float(tr.loc[is_kembar, "_y"].median()), 2),
            "p90_kembar": round(float(tr.loc[is_kembar, "_y"].quantile(0.9)), 2),
            "median_biasa": round(float(tr.loc[~is_kembar, "_y"].median()), 2),
            "p90_biasa": round(float(tr.loc[~is_kembar, "_y"].quantile(0.9)), 2),
        },
        "lebaran_2025": {
            "n_lebaran": int(is_lebaran.sum()),
            "median_lebaran": round(float(tr.loc[is_lebaran, "_y"].median()), 2),
            "p90_lebaran": round(float(tr.loc[is_lebaran, "_y"].quantile(0.9)), 2),
            "median_non_lebaran": round(float(tr.loc[~is_lebaran, "_y"].median()), 2),
            "p90_non_lebaran": round(float(tr.loc[~is_lebaran, "_y"].quantile(0.9)), 2),
        },
    }


def plot_pergeseran_temporal(train: pd.DataFrame, test: pd.DataFrame) -> Path:
    """Buat grafik 3-panel pergeseran temporal (Jan–Okt 2025) untuk laporan Tugas 1a.

    Panel (a): Pergeseran ``f35`` pada kurir ``K07`` mulai Juli 2025 dan kurir
    baru ``K13`` pada test dibandingkan kurir stabil (``K02``, ``K08``).
    Panel (b): Pertumbuhan ``antrean_hub_asal`` per hub beserta lonjakan Maret
    (Ramadan) dan Sep–Okt.
    Panel (c): Perbandingan ``f30`` (drift noise naik linear) vs ``f17`` (sinyal antrean).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.

    Returns
    -------
    Path
        Path file PNG yang tersimpan di ``kode/figures/task1_1a5_temporal_shifts.png``.
    """
    ensure_dirs()
    configure_plotting()

    pergeseran = cek_pergeseran_bulanan(train, test)
    piv_f35 = pergeseran["f35_per_kurir_bulan"].T
    piv_ant = pergeseran["antrean_per_hub_bulan"].T
    drift_f = pergeseran["ringkasan_drift_bulanan"]
    periode = list(drift_f.index)
    pos = np.arange(len(periode))
    # Bulan-bulan sesudah data latih (September dan Oktober) diarsir sebagai periode uji
    n_bulan_latih = pd.to_datetime(train["tgl_pickup"]).dt.to_period("M").nunique()
    rho = {c: float(train[c].corr(train["durasi_jam"], method="spearman")) for c in ("f30", "f17")}

    fig, axes = plt.subplots(1, 3, figsize=(LEBAR_GAMBAR, 2.0))

    # Panel (a): f35 kurir K07 (bergeser) dan K13 (baru) dibandingkan kurir yang stabil
    for k, col, style in [
        ("K07", "#dc2626", "-o"),
        ("K13", "#9333ea", "-s"),
        ("K02", "#f59e0b", "--^"),
        ("K08", "#16a34a", "--v"),
    ]:
        if k in piv_f35.columns:
            axes[0].plot(pos, piv_f35[k].reindex(periode), style, color=col, label=k)
    axes[0].set_title("(a) Median f35 per Kurir")
    axes[0].legend(loc="upper left", ncol=2, columnspacing=0.8)

    # Panel (b): Antrean per pusat sortir asal
    for hub, col in [("JKT", "#dc2626"), ("MKS", "#2563eb"), ("SBY", "#f97316"), ("BDG", "#16a34a")]:
        axes[1].plot(pos, piv_ant[hub].reindex(periode), "-o", color=col, label=hub)
    axes[1].set_title("(b) Median antrean_hub_asal")
    axes[1].legend(loc="upper right", ncol=2, columnspacing=0.8)

    # Panel (c): f30 (bergeser tanpa sinyal) dibandingkan f17 (bergeser dan bersinyal)
    axes[2].plot(pos, drift_f["f30"], "-o", color="#dc2626", label=f"f30 (\u03c1 = {angka_id(rho['f30'], 3)})")
    axes[2].plot(pos, drift_f["f17"], "-s", color="#2563eb", label=f"f17 (\u03c1 = {angka_id(rho['f17'], 3)})")
    axes[2].set_title("(c) Median f30 dan f17")
    axes[2].legend(loc="upper left")

    for ax in axes:
        ax.axvspan(n_bulan_latih - 0.5, len(periode) - 0.5, color="#9ca3af", alpha=0.25, linewidth=0)
        ax.set_xlim(-0.5, len(periode) - 0.5)
        ax.set_xticks(pos)
        # Label ditulis selang satu bulan agar muat tanpa diputar
        ax.set_xticklabels([label_bulan(m) if i % 2 == 0 else "" for i, m in enumerate(periode)])
        ax.grid(axis="x", visible=False)
        sumbu_koma(ax, "y")

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1a5_temporal_shifts.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


DAFTAR_CEK = (
    "skema",
    "duplikat",
    "id",
    "waktu",
    "tipe",
    "missing",
    "leakage",
    "target",
    "sentinel",
    "kategori",
    "temporal",
    "dampak",
)


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


def _tampilkan_target(train: pd.DataFrame, _test: pd.DataFrame) -> None:
    """Jalankan dan cetak hasil audit anomali target ``durasi_jam`` + grafik bukti."""
    info = cek_anomali_target(train)
    fig_path = plot_bukti_target_menit(train)
    print("=== [target] AUDIT TARGET durasi_jam & ERROR SATUAN MENIT ===")
    print(f"Baris <= 0 jam                : {info['n_le_zero']}")
    print(f"Kuantil mentah                : {info['kuantil_mentah']}")
    print(f"Baris > 1.000 jam             : {info['n_gt_ambang']} ({info['pct_gt_ambang']}%)")
    print(f"Celah bersih (<=1000 vs >1000): maks sah = {info['maks_sah_le_ambang']} jam | min ekstrem = {info['min_ekstrem_gt_ambang']}")
    print(f"Persentase kelipatan 6 (>1000): {info['pct_kelipatan_6']}% (0,1 jam x 60 = 6)")
    print(f"Baris mencurigakan <= 1000    : {info['suspicious_le_ambang_low_scan']}")
    print(f"Mean target (mentah -> /60)   : {info['mean_mentah']} jam -> {info['mean_setelah_koreksi']} jam")
    print(f"Median bersih vs ekstrem / 60 : {info['median_bersih']} jam vs {info['median_ekstrem_dibagi_60']} jam")
    print(f"Median jam/scan (bersih vs/60): {info['median_jam_per_scan_bersih']} vs {info['median_jam_per_scan_koreksi']}")
    print(f"Grafik bukti tersimpan di     : {fig_path.relative_to(KODE_DIR.parent)}")


def _tampilkan_sentinel(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak pemeriksaan batas fisik numerik & pemulihan sentinel ``jarak_km``."""
    tabel = cek_batas_fisik_numerik(train, test)
    rute = cek_pemulihan_rute_jarak(train, test)
    print("=== [sentinel] CEK BATAS FISIK FITUR NUMERIK BERNAMA ===")
    print(tabel.to_string(index=False))
    print("\n=== [sentinel] DETAIL SENTINEL jarak_km & PEMULIHAN RUTE ===")
    print(
        f"Train rusak (0 / 9999 / total): {rute['train_nol']} / {rute['train_9999']} / "
        f"{rute['train_total_rusak']} ({rute['train_pct_rusak']}%)"
    )
    print(
        f"Test rusak  (0 / 9999 / total): {rute['test_nol']} / {rute['test_9999']} / "
        f"{rute['test_total_rusak']} ({rute['test_pct_rusak']}%)"
    )
    print(
        f"Konsistensi rute valid train  : {rute['jumlah_rute_valid_train']:,} rute | "
        f"max_nunique={rute['max_nunique_per_rute']} | max_std={rute['max_std_per_rute']}"
    )
    print(
        f"Test rusak pulih eksak (train): {rute['test_rusak_terpulihkan_eksak']}/{rute['test_total_rusak']} "
        f"({rute['test_rusak_terpulihkan_pct']}%)"
    )


def _tampilkan_kategori(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak audit 6 kolom kategorikal beserta cek hierarki."""
    info = cek_kolom_kategorikal(train, test)
    print("=== [kategori] AUDIT 6 KOLOM KATEGORIKAL & UNSEEN TEST ===")
    print(info["ringkasan_kategori"].to_string(index=False))
    print(
        f"\nHierarki 1-ke-1: max hub per kota_asal = {info['max_hub_per_kota_asal']} | "
        f"max pulau per kota_tujuan = {info['max_pulau_per_kota_tujuan']}"
    )
    print(
        f"Sebaran kota_tujuan (train): min baris per kota = {info['min_baris_kota_tujuan_train']} | "
        f"kota dengan <50 baris = {info['kota_tujuan_kurang_50_baris']}"
    )


def _tampilkan_temporal(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Jalankan dan cetak audit pergeseran bulanan, efek kalender, dan grafik 3-panel."""
    geser = cek_pergeseran_bulanan(train, test)
    kal = cek_efek_kalender(train)
    fig_path = plot_pergeseran_temporal(train, test)

    print("=== [temporal] PERGESERAN BULANAN (JAN–OKT 2025) ===")
    print("\n1. Median f35 untuk kurir kunci (K07 bergeser di Juli; K13 baru muncul di Sep–Okt):")
    print(geser["f35_per_kurir_bulan"].loc[["K01", "K02", "K07", "K08", "K13"]].to_string())

    print("\n2. Median durasi_jam terkoreksi untuk kurir kunci pada train (Jan–Agu):")
    print(geser["durasi_per_kurir_bulan"].loc[["K01", "K02", "K07", "K08"]].to_string())

    print("\n3. Median antrean_hub_asal per hub x bulan (Jan–Okt):")
    print(geser["antrean_per_hub_bulan"].to_string())

    print("\n4. Efek kalender pada durasi_jam terkoreksi (train):")
    print(kal["per_hari"].to_string())
    tk = kal["tanggal_kembar"]
    lb = kal["lebaran_2025"]
    print(
        f"   Tanggal kembar (n={tk['n_kembar']:,}): median={tk['median_kembar']} jam (P90={tk['p90_kembar']}) "
        f"vs hari biasa: median={tk['median_biasa']} jam (P90={tk['p90_biasa']})"
    )
    print(
        f"   Lebaran 26 Mar–6 Apr (n={lb['n_lebaran']:,}): median={lb['median_lebaran']} jam (P90={lb['p90_lebaran']}) "
        f"vs non-Lebaran: median={lb['median_non_lebaran']} jam (P90={lb['p90_non_lebaran']})"
    )
    print(f"\nGrafik bukti tersimpan di: {fig_path.relative_to(KODE_DIR.parent)}")


def _tampilkan_dampak(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Jalankan dan cetak hasil pengukuran dampak kuantitatif Langkah 1a.6b."""
    hasil = jalankan_semua_dampak(train, test)
    e1 = hasil["exp1_leakage_jumlah_scan"]
    e2 = hasil["exp2_satuan_target"]
    e3 = hasil["exp3_pemulihan_jarak_km"]
    e4 = hasil["exp4_imputasi_berat_kg"]
    e5 = hasil["exp5_seleksi_fitur_f"]

    print("=== [dampak] PENGUKURAN DAMPAK KUANTITATIF (TRAIN JAN–JUN -> VAL JUL–AGU) ===")
    print("\n1. Dampak Leakage jumlah_scan (LightGBM Quantile):")
    for nama, m in e1.items():
        print(f"   {nama:28s} -> P50={m['pinball_p50']:.4f} | P90={m['pinball_p90']:.4f} | cov90={m['p90_coverage']:.4f}")

    print("\n2. Dampak Error Satuan Target durasi_jam > 1000 (370 baris x60):")
    r_raw = e2["ridge_target_mentah"]
    r_cln = e2["ridge_target_koreksi_60"]
    l_raw = e2["lgbm_target_mentah"]
    l_cln = e2["lgbm_target_koreksi_60"]
    print(f"   Ridge (target mentah) -> RMSE={r_raw['rmse']:.2f} | MAE={r_raw['mae']:.2f} | bias={r_raw['mean_bias']:+.2f} jam")
    print(f"   Ridge (target / 60)   -> RMSE={r_cln['rmse']:.2f} | MAE={r_cln['mae']:.2f} | bias={r_cln['mean_bias']:+.2f} jam")
    print(f"   LGBM  (target mentah) -> P50={l_raw['pinball_p50']:.4f} | P90={l_raw['pinball_p90']:.4f} | cov90={l_raw['p90_coverage']:.4f}")
    print(f"   LGBM  (target / 60)   -> P50={l_cln['pinball_p50']:.4f} | P90={l_cln['pinball_p90']:.4f} | cov90={l_cln['p90_coverage']:.4f}")

    print(f"\n3. Dampak Pemulihan Sentinel jarak_km (subset sentinel val n={e3['n_sentinel_validasi']}):")
    s_raw = e3["subset_sentinel_mentah"]
    s_rec = e3["subset_sentinel_dipulihkan"]
    print(f"   Sentinel mentah     -> P50={s_raw['pinball_p50']:.4f} | P90={s_raw['pinball_p90']:.4f} | MAE={s_raw['mae_p50']:.2f} jam")
    print(f"   Sentinel dipulihkan -> P50={s_rec['pinball_p50']:.4f} | P90={s_rec['pinball_p90']:.4f} | MAE={s_rec['mae_p50']:.2f} jam")

    print("\n4. Dampak Imputasi berat_kg (MAE kg pada validasi):")
    print(f"   Global (median vs volume*densitas): {e4['global_mae_median_kg']:.3f} kg -> {e4['global_mae_densitas_kg']:.3f} kg")
    print(f"   Kargo  (median vs volume*densitas): {e4['kargo_mae_median_kg']:.3f} kg -> {e4['kargo_mae_densitas_kg']:.3f} kg")

    print("\n5. Dampak Memangkas 40 Fitur Noise f (termasuk f30 drift):")
    f50 = e5["semua_50_fitur_f"]
    f10 = e5["hanya_10_fitur_sinyal_f"]
    print(f"   Semua 50 fitur f -> P50={f50['pinball_p50']:.4f} | P90={f50['pinball_p90']:.4f} | cov90={f50['p90_coverage']:.4f}")
    print(f"   10 fitur sinyal  -> P50={f10['pinball_p50']:.4f} | P90={f10['pinball_p90']:.4f} | cov90={f10['p90_coverage']:.4f}")
    print(f"\nGrafik dampak tersimpan di: kode/figures/{hasil['grafik_dampak']}")
    return hasil


def simpan_ringkasan_audit_1a(
    train: pd.DataFrame,
    test: pd.DataFrame,
    hasil_dampak: dict[str, Any] | None = None,
) -> Path:
    """Rangkum dan simpan seluruh temuan Tugas 1a ke ``kode/results/task1_audit.json``.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih regresi.
    test : pd.DataFrame
        Data uji regresi.
    hasil_dampak : dict[str, Any] or None, default=None
        Jika sudah dihitung, gunakan hasil dampak tersebut agar tidak dihitung ulang.

    Returns
    -------
    Path
        Path file JSON yang tersimpan.
    """
    ensure_dirs()
    tabel_scr = screening_kolom_numerik(train, test)
    kat = cek_kolom_kategorikal(train, test)
    kal = cek_efek_kalender(train)
    dampak = hasil_dampak if hasil_dampak is not None else jalankan_semua_dampak(train, test)

    # Statistik bukti (KS train-vs-test, Spearman ke target, median/maks) untuk kolom
    # yang bukan "stabil", sebagai rincian tabel temuan audit.
    kuadran = kelompokkan_kuadran_detektif(tabel_scr)
    kolom_kunci = kuadran["tersangka_leakage"] + kuadran["drift_bersinyal"] + kuadran["drift_noise"]
    statistik_kunci = tabel_scr.set_index("kolom").loc[kolom_kunci].to_dict(orient="index")

    ringkasan = {
        "task": "Tugas 1a — Audit Data & Pengukuran Dampak (Regresi)",
        "seed": SEED,
        "langkah_1a1_skema_waktu": audit_skema_dan_waktu(train, test),
        "langkah_1a2_tipe_missing": audit_tipe_dan_missing(train, test),
        "langkah_1a3_kuadran_leakage_drift": {**kuadran, "statistik_kolom_kunci": statistik_kunci},
        "langkah_1a4_target_dan_jarak": {
            "target": cek_anomali_target(train),
            "pemulihan_jarak": cek_pemulihan_rute_jarak(train, test),
        },
        "langkah_1a5_kategori_dan_kalender": {
            "kategori": kat["ringkasan_kategori"].to_dict(orient="records"),
            "hierarki": {
                "max_hub_per_kota_asal": kat["max_hub_per_kota_asal"],
                "max_pulau_per_kota_tujuan": kat["max_pulau_per_kota_tujuan"],
                "min_baris_kota_tujuan_train": kat["min_baris_kota_tujuan_train"],
                "kota_tujuan_kurang_50_baris": kat["kota_tujuan_kurang_50_baris"],
            },
            "tanggal_kembar": kal["tanggal_kembar"],
            "lebaran_2025": kal["lebaran_2025"],
        },
        "langkah_1a6_pengukuran_dampak": dampak,
    }
    out_path = RESULTS_DIR / "task1_audit.json"
    save_json(ringkasan, out_path)
    return out_path


def main() -> None:
    """Jalankan satu, beberapa, atau seluruh pemeriksaan audit Tugas 1a lewat CLI.

    Contoh penggunaan:
        python kode/src/regresi/audit.py                 # Jalankan semua + ekspor JSON
        python kode/src/regresi/audit.py --cek dampak    # Hanya 5 eksperimen ukur dampak
        python kode/src/regresi/audit.py --cek target    # Hanya cek anomali target
    """
    set_global_seed(SEED)
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
    parser.add_argument(
        "--simpan-json",
        action="store_true",
        help="Simpan ringkasan lengkap Tugas 1a ke kode/results/task1_audit.json.",
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
        "target": _tampilkan_target,
        "sentinel": _tampilkan_sentinel,
        "kategori": _tampilkan_kategori,
        "temporal": _tampilkan_temporal,
        "dampak": _tampilkan_dampak,
    }

    hasil_dampak: dict[str, Any] | None = None
    for i, nama in enumerate(args.cek):
        if i > 0:
            print()
        ret = handlers[nama](train, test)
        if nama == "dampak":
            hasil_dampak = ret

    # Ekspor JSON otomatis saat menjalankan seluruh cek atau bila --simpan-json diaktifkan
    if args.simpan_json or set(args.cek) == set(DAFTAR_CEK):
        json_path = simpan_ringkasan_audit_1a(train, test, hasil_dampak=hasil_dampak)
        print(f"\nRingkasan lengkap Tugas 1a tersimpan di: {json_path.relative_to(KODE_DIR.parent)}")


if __name__ == "__main__":
    main()
