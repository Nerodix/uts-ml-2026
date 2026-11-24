"""Konstanta skema, pembersihan data, dan prapemrosesan anti-leakage Tugas 2 (Klasifikasi Fraud).

Modul ini mendefinisikan daftar kolom resmi hasil audit forensik Tugas 2a serta
transformator prapemrosesan yang 100% kausal (seluruh fitur tersedia saat
transaksi terjadi, sebelum tinjauan manual dan sebelum konfirmasi fraud):
- Membuang fitur leakage pasca-transaksi ``flag_review_manual``.
- Memisahkan 8 fitur ``v`` bersinyal dari 42 fitur noise (termasuk ``v15`` yang
  mengalami *drift* linear terhadap waktu tanpa sinyal fraud).
- Menambahkan indikator kualitas & pola domain baris-tunggal yang tersedia
  seketika saat transaksi (``is_trx_pertama_pengguna``, ``is_umur_merchant_negatif``,
  ``is_persen_saldo_capped``, ``is_nominal_bulat_100k``, ``is_nominal_kecil_25k``).
- Menyelaraskan tipe kategorikal secara deterministik dan aman terhadap kategori
  *unseen* di data uji (kompatibel dengan Pandas 3.x).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import KLASIFIKASI_TEST_PATH, KLASIFIKASI_TRAIN_PATH, SNAPSHOT_DATE

ID_COLUMN = "id_transaksi"
TIME_COLUMN = "waktu"
USER_ID_COLUMN = "id_pengguna"
MERCHANT_ID_COLUMN = "id_merchant"
TARGET_COLUMN = "is_fraud"

# Tanggal penarikan snapshot data latih sesuai kamus_data.md
SNAPSHOT_TIMESTAMP = pd.Timestamp(f"{SNAPSHOT_DATE} 23:59:59")
# Batas akhir periode label matang (sebelum jendela right-censoring Agustus)
MATURE_TRAIN_END_DATE = "2025-07-31"
# Awal gelombang pola fraud QRIS baru yang muncul pada awal Agustus 2025
QRIS_SHIFT_START_DATE = "2025-08-01"
# Batas tanggal awal Agustus yang relatif matang (~21-31 hari sebelum snapshot)
EARLY_AUG_MATURE_END = "2025-08-10"

# Fitur bocor pasca-transaksi (ditetapkan setelah peninjauan/laporan risiko)
LEAKAGE_COLUMNS: tuple[str, ...] = ("flag_review_manual",)

# Kolom kategorikal transaksi (kardinalitas rendah-menengah, 0 unseen di test)
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "kategori_merchant",
    "kanal",
    "tipe_perangkat",
    "email_domain",
)

# 25 fitur numerik bernama yang tersedia saat transaksi terjadi
NAMED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "nominal",
    "jam",
    "hari_libur",
    "umur_akun_hari",
    "level_kyc",
    "umur_pengguna_tahun",
    "umur_perangkat_hari",
    "perangkat_baru",
    "jumlah_akun_per_perangkat",
    "ganti_password_7h",
    "gagal_login_24j",
    "jarak_lokasi_km",
    "negara_ip_beda",
    "kartu_tidak_hadir",
    "menit_sejak_trx_terakhir",
    "trx_1j",
    "trx_24j",
    "trx_7h",
    "nominal_24j",
    "rasio_nominal_rata2",
    "saldo_sebelum",
    "persen_saldo",
    "umur_merchant_hari",
    "refund_merchant_30h",
    "volume_merchant_30h",
)

ANONYMOUS_V_COLUMNS: tuple[str, ...] = tuple(f"v{i:02d}" for i in range(1, 51))

# 8 fitur anonim dengan sinyal nyata (teruji univariat, non-linear, residual, & pergeseran Agustus):
# v03 (ATO/perangkat+lokasi), v11 (kecepatan trx), v19 (refund merchant), v26 (nominal/gelombang QRIS),
# v29 (sinyal independen murni), v38 (umur akun/perangkat), v42 (pola jam), v47 (akun per perangkat).
SIGNAL_V_COLUMNS: tuple[str, ...] = ("v03", "v11", "v19", "v26", "v29", "v38", "v42", "v47")
DRIFT_NOISE_V_COLUMNS: tuple[str, ...] = ("v15",)
NOISE_V_COLUMNS: tuple[str, ...] = tuple(c for c in ANONYMOUS_V_COLUMNS if c not in SIGNAL_V_COLUMNS)

# Indikator kualitas & pola baris-tunggal yang tersedia seketika saat transaksi
QUALITY_FLAG_COLUMNS: tuple[str, ...] = (
    "is_trx_pertama_pengguna",
    "is_umur_merchant_negatif",
    "is_persen_saldo_capped",
    "is_nominal_bulat_100k",
    "is_nominal_bulat_50k",
    "is_nominal_kecil_25k",
)

# Himpunan fitur dasar bersih (bebas leakage, bebas 42 noise v) untuk audit & validasi 2a–2b
CLEAN_BASE_FEATURES: tuple[str, ...] = (
    CATEGORICAL_COLUMNS + NAMED_NUMERIC_COLUMNS + SIGNAL_V_COLUMNS
)

# Himpunan fitur bersih + indikator kualitas baris-tunggal
CLEAN_WITH_FLAGS_FEATURES: tuple[str, ...] = CLEAN_BASE_FEATURES + QUALITY_FLAG_COLUMNS

# Batas bin hari sebelum snapshot untuk memantau kurva maturitas konfirmasi label
MATURITY_DAY_BINS: tuple[float, ...] = (0.0, 3.0, 7.0, 10.0, 14.0, 18.0, 21.0, 25.0, 31.0, 45.0)


def muat_data_klasifikasi(
    train_path: Path = KLASIFIKASI_TRAIN_PATH,
    test_path: Path = KLASIFIKASI_TEST_PATH,
    parse_waktu: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Muat data latih dan data uji klasifikasi fraud beserta kolom waktu terurai.

    Parameters
    ----------
    train_path : Path
        Lokasi berkas ``klasifikasi_train.csv.gz``.
    test_path : Path
        Lokasi berkas ``klasifikasi_test.csv.gz``.
    parse_waktu : bool, default=True
        Bila ``True``, menambahkan kolom ``waktu_dt`` bertipe ``datetime64``
        (tanpa mengubah urutan baris asli data uji/latih).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Pasangan ``(train, test)`` dalam urutan baris asli berkas sumber.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if parse_waktu:
        train["waktu_dt"] = pd.to_datetime(train[TIME_COLUMN])
        test["waktu_dt"] = pd.to_datetime(test[TIME_COLUMN])
    return train, test


def urutkan_kronologis(df: pd.DataFrame) -> pd.DataFrame:
    """Urutkan transaksi secara deterministik berdasarkan ``(waktu_dt, id_transaksi)``.

    Data mentah ``klasifikasi_train`` dan ``klasifikasi_test`` tidak terurut
    kronologis. Pengurutan stabil wajib dilakukan sebelum menghitung fitur
    riwayat kausal pengguna/merchant (Tugas 2c).

    Parameters
    ----------
    df : pd.DataFrame
        Data transaksi yang memuat ``waktu`` (atau ``waktu_dt``) dan ``id_transaksi``.

    Returns
    -------
    pd.DataFrame
        Salinan DataFrame terurut kronologis dengan indeks asli dipertahankan
        pada kolom ``_orig_idx`` bila perlu pemetaan balik.
    """
    out = df.copy()
    if "waktu_dt" not in out.columns:
        out["waktu_dt"] = pd.to_datetime(out[TIME_COLUMN])
    return out.sort_values(["waktu_dt", ID_COLUMN], kind="mergesort")


def tambah_flag_kualitas_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan indikator kualitas data dan pola nominal baris-tunggal bebas leakage.

    Seluruh indikator hanya membaca atribut baris transaksi saat kejadian:
    - ``is_trx_pertama_pengguna`` : 1 bila ``menit_sejak_trx_terakhir`` bernilai NaN
      (transaksi pertama pengguna di log; laju fraud 0,94% vs 0,58%).
    - ``is_umur_merchant_negatif``: 1 bila ``umur_merchant_hari < 0`` (anomali tanggal
      pendaftaran merchant dengan laju fraud 20,31%, ~31x rata-rata).
    - ``is_persen_saldo_capped``  : 1 bila ``persen_saldo`` menyentuh batas atas 50,0
      (transaksi menguras >=50x saldo dompet; laju fraud 45,12%).
    - ``is_nominal_bulat_100k``   : 1 bila ``nominal`` kelipatan persis Rp 100.000
      (muncul pada 79,3% gelombang fraud QRIS Agustus vs 0,1% transaksi normal).
    - ``is_nominal_bulat_50k``    : 1 bila ``nominal`` kelipatan persis Rp 50.000.
    - ``is_nominal_kecil_25k``    : 1 bila ``nominal <= Rp 25.000`` (di bawah biaya
      tinjauan manual; khas pola *card testing*).

    Parameters
    ----------
    df : pd.DataFrame
        Data transaksi mentah atau terfilter.

    Returns
    -------
    pd.DataFrame
        Salinan DataFrame dengan 6 kolom indikator biner bertipe ``int8``.
    """
    out = df.copy()
    nom = out["nominal"].to_numpy(dtype=np.float64)
    out["is_trx_pertama_pengguna"] = out["menit_sejak_trx_terakhir"].isna().astype(np.int8)
    out["is_umur_merchant_negatif"] = (out["umur_merchant_hari"].to_numpy(dtype=np.float64) < 0.0).astype(
        np.int8
    )
    out["is_persen_saldo_capped"] = (out["persen_saldo"].to_numpy(dtype=np.float64) >= 49.999).astype(
        np.int8
    )
    out["is_nominal_bulat_100k"] = (np.mod(nom, 100_000.0) == 0.0).astype(np.int8)
    out["is_nominal_bulat_50k"] = (np.mod(nom, 50_000.0) == 0.0).astype(np.int8)
    out["is_nominal_kecil_25k"] = (nom <= 25_000.0).astype(np.int8)
    return out


def samakan_tipe_kategori(
    train_df: pd.DataFrame,
    other_dfs: Sequence[pd.DataFrame],
    cat_cols: Sequence[str] = CATEGORICAL_COLUMNS,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Selaraskan kategori kolom kategorikal dari ``train_df`` ke DataFrame lain.

    Menggunakan konstruktor ``pd.Categorical`` agar nilai kategori baru (*unseen*)
    pada validasi/uji otomatis dipetakan ke ``NaN`` tanpa memicu ``Pandas4Warning``
    pada Pandas 3.x.

    Parameters
    ----------
    train_df : pd.DataFrame
        Data latih acuan daftar kategori.
    other_dfs : Sequence[pd.DataFrame]
        Daftar DataFrame validasi/uji yang kategori kolomnya diselaraskan.
    cat_cols : Sequence[str], default=CATEGORICAL_COLUMNS
        Daftar kolom kategorikal yang diproses.

    Returns
    -------
    tuple[pd.DataFrame, list[pd.DataFrame]]
        ``(train_aligned, others_aligned)`` dengan tipe ``category`` identik.
    """
    tr = train_df.copy()
    others = [d.copy() for d in other_dfs]
    for col in cat_cols:
        if col not in tr.columns:
            continue
        cats = pd.Index(tr[col].astype("string").dropna().unique()).sort_values()
        tr[col] = pd.Categorical(tr[col].astype("string"), categories=cats, ordered=False)
        for df_other in others:
            if col in df_other.columns:
                df_other[col] = pd.Categorical(
                    df_other[col].astype("string"), categories=cats, ordered=False
                )
    return tr, others


def hitung_hari_sebelum_snapshot(
    waktu_dt: pd.Series,
    snapshot_ts: pd.Timestamp = SNAPSHOT_TIMESTAMP,
) -> pd.Series:
    """Hitung selisih hari (kontinu) dari waktu transaksi hingga tanggal penarikan snapshot.

    Parameters
    ----------
    waktu_dt : pd.Series
        Seri stempel waktu transaksi bertipe ``datetime64``.
    snapshot_ts : pd.Timestamp, default=SNAPSHOT_TIMESTAMP
        Stempel waktu penarikan data latih (``2025-08-31 23:59:59``).

    Returns
    -------
    pd.Series
        Jarak hari positif menuju snapshot; nilai kecil menandakan transaksi baru
        yang rentan terhadap *right-censoring* (konfirmasi fraud belum selesai).
    """
    return (snapshot_ts - pd.to_datetime(waktu_dt)).dt.total_seconds() / 86_400.0


def estimasi_kurva_maturitas_label(
    train_df: pd.DataFrame,
    snapshot_ts: pd.Timestamp = SNAPSHOT_TIMESTAMP,
    bins: Sequence[float] = MATURITY_DAY_BINS,
) -> pd.DataFrame:
    """Estimasi kurva empiris maturitas konfirmasi label berdasarkan jarak ke snapshot.

    Membandingkan laju fraud teramati pada setiap rentang jarak hari sebelum
    snapshot (khusus transaksi non-QRIS yang lajunya stabil sepanjang Jan–Jul)
    terhadap baseline matang Jan–Jul 2025. Pemisahan non-QRIS mencegah bias dari
    gelombang fraud QRIS baru yang muncul bersamaan pada Agustus 2025.

    Parameters
    ----------
    train_df : pd.DataFrame
        Data latih klasifikasi yang memuat ``waktu``/``waktu_dt``, ``kanal``, dan ``is_fraud``.
    snapshot_ts : pd.Timestamp, default=SNAPSHOT_TIMESTAMP
        Stempel waktu penarikan snapshot.
    bins : Sequence[float], default=MATURITY_DAY_BINS
        Batas rentang hari sebelum snapshot.

    Returns
    -------
    pd.DataFrame
        Tabel maturitas per bin jarak hari dengan kolom: ``rentang_hari``, ``tengah_hari``,
        ``n_non_qris``, ``fraud_non_qris``, ``laju_non_qris``, ``rasio_maturitas``,
        ``n_qris``, ``fraud_qris``, dan ``laju_qris``.
    """
    df = train_df.copy()
    if "waktu_dt" not in df.columns:
        df["waktu_dt"] = pd.to_datetime(df[TIME_COLUMN])
    df["hari_sebelum_snapshot"] = hitung_hari_sebelum_snapshot(df["waktu_dt"], snapshot_ts)

    # Baseline matang non-QRIS dihitung pada periode Jan–Jul (jarak > 31 hari)
    baseline_non_qris = float(
        df.loc[(df["waktu_dt"] <= pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")) & (df["kanal"] != "qris"), TARGET_COLUMN].mean()
    )

    sub = df[df["hari_sebelum_snapshot"] <= max(bins)].copy()
    sub["bin"] = pd.cut(sub["hari_sebelum_snapshot"], bins=bins, include_lowest=True)

    rows: list[dict[str, float | int | str]] = []
    for interval, grp in sub.groupby("bin", observed=False):
        non_q = grp[grp["kanal"] != "qris"]
        qris = grp[grp["kanal"] == "qris"]
        laju_nq = float(non_q[TARGET_COLUMN].mean()) if len(non_q) > 0 else 0.0
        laju_q = float(qris[TARGET_COLUMN].mean()) if len(qris) > 0 else 0.0
        rasio = float(np.clip(laju_nq / baseline_non_qris, 0.0, 1.0)) if baseline_non_qris > 0 else 0.0
        lo = int(round(interval.left))
        hi = int(round(interval.right))
        rows.append(
            {
                "rentang_hari": f"{lo}–{hi}h",
                "hari_min": float(interval.left),
                "hari_max": float(interval.right),
                "tengah_hari": 0.5 * (float(interval.left) + float(interval.right)),
                "n_non_qris": int(len(non_q)),
                "fraud_non_qris": int(non_q[TARGET_COLUMN].sum()),
                "laju_non_qris": laju_nq,
                "rasio_maturitas": rasio,
                "n_qris": int(len(qris)),
                "fraud_qris": int(qris[TARGET_COLUMN].sum()),
                "laju_qris": laju_q,
                "baseline_non_qris": baseline_non_qris,
            }
        )
    return pd.DataFrame(rows)
