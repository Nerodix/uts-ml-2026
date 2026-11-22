"""Modul pembersihan data anti-leakage untuk Tugas 1 (Regresi Estimasi Durasi).

Seluruh transformasi statistik di modul ini mengikuti pola ``fit()`` pada data
latih saja lalu ``transform()`` ke data validasi/uji agar tidak ada kebocoran
informasi antar split:
1. Koreksi target ``durasi_jam`` > 1.000 jam (salah satuan menit -> dibagi 60).
2. Pemulihan sentinel ``jarak_km`` (0 dan 9999) dari kamus rute data latih.
3. Imputasi ``berat_kg`` berbasis densitas ``berat_kg / volume_liter`` per layanan.
4. Penyelarasan tipe kategori agar kategori *unseen* di test (``K13``) aman.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_COLUMN = "id_paket"
DATE_COLUMN = "tgl_pickup"
TARGET_COLUMN = "durasi_jam"
LEAKAGE_COLUMNS = ("jumlah_scan",)

CATEGORICAL_COLUMNS = (
    "layanan",
    "mitra_kurir",
    "hub_asal",
    "kota_asal",
    "pulau_tujuan",
    "kota_tujuan",
)

NAMED_NUMERIC_COLUMNS = (
    "jam_pickup",
    "jarak_km",
    "berat_kg",
    "volume_liter",
    "jumlah_item",
    "nilai_barang",
    "asuransi",
    "cod",
    "alamat_skor",
    "pengalaman_kurir_bulan",
    "rating_penjual",
    "umur_akun_penjual_hari",
    "volume_penjual_30h",
    "antrean_hub_asal",
    "indeks_kemacetan",
    "hujan_asal_mm",
    "hujan_tujuan_mm",
)

ANONYMOUS_F_COLUMNS = tuple(f"f{i:02d}" for i in range(1, 51))
SIGNAL_F_COLUMNS = ("f04", "f05", "f09", "f13", "f17", "f22", "f23", "f28", "f35", "f41")

TARGET_UNIT_THRESHOLD_HOURS = 1000.0
JARAK_SENTINEL_LOW = 0.0
JARAK_SENTINEL_HIGH = 9999.0


def fix_target_unit_errors(
    y: pd.Series | np.ndarray,
    threshold: float = TARGET_UNIT_THRESHOLD_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Koreksi target ``durasi_jam`` yang salah tercatat dalam satuan menit (``* 60``).

    Berdasarkan audit Langkah 1a.4, seluruh 370 baris dengan ``durasi_jam > 1000``
    merupakan kelipatan eksak 6 dengan celah bersih dari nilai sah maksimum (693,1 jam).

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Vektor target ``durasi_jam`` mentah.
    threshold : float, default=1000.0
        Ambang batas (jam) pemisah satuan jam vs menit.

    Returns
    -------
    y_clean : np.ndarray
        Target ``durasi_jam`` dalam satuan jam (``float64``).
    corrupt_mask : np.ndarray
        Masker boolean (``True`` pada baris yang dibagi 60).
    """
    arr = np.asarray(y, dtype=np.float64)
    corrupt_mask = arr > threshold
    y_clean = np.where(corrupt_mask, arr / 60.0, arr)
    return y_clean, corrupt_mask


def identify_corrupt_jarak_mask(jarak: pd.Series | np.ndarray) -> np.ndarray:
    """Identifikasi baris dengan nilai sentinel ``jarak_km`` rusak (<= 0 atau >= 9999).

    Parameters
    ----------
    jarak : pd.Series or np.ndarray
        Nilai kolom ``jarak_km`` mentah.

    Returns
    -------
    np.ndarray
        Masker boolean dengan nilai ``True`` pada baris sentinel/rusak.
    """
    arr = np.asarray(jarak, dtype=np.float64)
    return np.isnan(arr) | (arr <= JARAK_SENTINEL_LOW) | (arr >= JARAK_SENTINEL_HIGH)


@dataclass
class RouteDistanceImputer:
    """Pemulih sentinel ``jarak_km`` menggunakan kamus rute hierarkis dari data latih.

    Catatan Anti-Leakage
    --------------------
    Kamus jarak ``(kota_asal, kota_tujuan)`` beserta fallback ``(hub_asal, pulau_tujuan)``,
    ``pulau_tujuan``, dan median global hanya dipelajari saat ``fit(train_df)``
    pada baris valid, tanpa pernah membaca data validasi atau data uji.
    """

    route_map: dict[tuple[str, str], float] = field(default_factory=dict)
    hub_island_map: dict[tuple[str, str], float] = field(default_factory=dict)
    island_map: dict[str, float] = field(default_factory=dict)
    global_median: float = 0.0

    def fit(self, df: pd.DataFrame) -> RouteDistanceImputer:
        """Pelajari kamus jarak rute hanya dari baris ``jarak_km`` valid di data latih.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame latih.

        Returns
        -------
        RouteDistanceImputer
            Objek imputer yang sudah ter-fit.
        """
        valid = df.loc[~identify_corrupt_jarak_mask(df["jarak_km"])]
        self.route_map = (
            valid.groupby(["kota_asal", "kota_tujuan"])["jarak_km"].median().to_dict()
        )
        self.hub_island_map = (
            valid.groupby(["hub_asal", "pulau_tujuan"])["jarak_km"].median().to_dict()
        )
        self.island_map = valid.groupby("pulau_tujuan")["jarak_km"].median().to_dict()
        self.global_median = float(valid["jarak_km"].median())
        return self

    def transform(self, df: pd.DataFrame, add_indicator: bool = True) -> pd.DataFrame:
        """Pulihkan nilai sentinel ``jarak_km`` menggunakan kamus hasil ``fit()``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame target (latih, validasi, atau uji).
        add_indicator : bool, default=True
            Jika ``True``, tambahkan kolom biner ``jarak_km_was_corrupt``.

        Returns
        -------
        pd.DataFrame
            Salinan DataFrame dengan ``jarak_km`` terpulihkan.
        """
        out = df.copy()
        bad_mask = identify_corrupt_jarak_mask(out["jarak_km"])
        if add_indicator:
            out["jarak_km_was_corrupt"] = bad_mask.astype(np.int8)

        if not bad_mask.any():
            return out

        bad_idx = out.index[bad_mask]
        recovered = np.empty(len(bad_idx), dtype=np.float64)
        ka = out.loc[bad_idx, "kota_asal"].to_numpy()
        kt = out.loc[bad_idx, "kota_tujuan"].to_numpy()
        hub = out.loc[bad_idx, "hub_asal"].to_numpy()
        pul = out.loc[bad_idx, "pulau_tujuan"].to_numpy()

        for i in range(len(bad_idx)):
            val = self.route_map.get((ka[i], kt[i]))
            if val is None:
                val = self.hub_island_map.get((hub[i], pul[i]))
            if val is None:
                val = self.island_map.get(pul[i], self.global_median)
            recovered[i] = val

        out.loc[bad_idx, "jarak_km"] = recovered
        return out


@dataclass
class BeratImputer:
    """Imputer ``berat_kg`` berbasis hubungan fisik ``berat_kg ~ densitas * volume_liter``.

    Catatan Anti-Leakage
    --------------------
    Median densitas per ``layanan`` hanya dihitung pada baris non-missing di
    ``fit(train_df)`` sehingga aman untuk validasi waktu dan data uji.
    """

    density_by_service: dict[str, float] = field(default_factory=dict)
    global_density: float = 0.25

    def fit(self, df: pd.DataFrame) -> BeratImputer:
        """Hitung median densitas ``berat_kg / volume_liter`` per layanan pada data latih.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame latih.

        Returns
        -------
        BeratImputer
            Objek imputer yang sudah ter-fit.
        """
        valid = df.loc[df["berat_kg"].notna() & (df["volume_liter"] > 0)].copy()
        valid["_dens"] = valid["berat_kg"] / valid["volume_liter"]
        self.density_by_service = valid.groupby("layanan")["_dens"].median().to_dict()
        self.global_density = float(valid["_dens"].median())
        return self

    def transform(self, df: pd.DataFrame, add_indicator: bool = True) -> pd.DataFrame:
        """Imputasi ``berat_kg`` yang kosong dan opsional tambahkan indikator biner.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame target (latih, validasi, atau uji).
        add_indicator : bool, default=True
            Jika ``True``, tambahkan kolom biner ``berat_kg_missing``.

        Returns
        -------
        pd.DataFrame
            Salinan DataFrame dengan ``berat_kg`` terimputasi.
        """
        out = df.copy()
        miss_mask = out["berat_kg"].isna()
        if add_indicator:
            out["berat_kg_missing"] = miss_mask.astype(np.int8)

        if not miss_mask.any():
            return out

        dens = out.loc[miss_mask, "layanan"].map(self.density_by_service).fillna(self.global_density)
        out.loc[miss_mask, "berat_kg"] = (out.loc[miss_mask, "volume_liter"] * dens).astype(np.float64)
        return out


def align_categorical_dtypes(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    cat_cols: tuple[str, ...] = CATEGORICAL_COLUMNS,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Samakan ``CategoricalDtype`` antara data latih dan data validasi/uji.

    Kategori yang hanya muncul di data uji (mis. ``mitra_kurir == 'K13'``)
    otomatis dipetakan ke ``NaN`` sehingga model pohon menanganinya konsisten
    tanpa error maupun kebocoran kategori dari test.

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame latih sebagai acuan daftar kategori sah.
    other_dfs : list[pd.DataFrame]
        Daftar DataFrame validasi/uji yang akan diselaraskan.
    cat_cols : tuple[str, ...], default=CATEGORICAL_COLUMNS
        Kolom kategorikal yang dikonversi ke tipe ``category``.

    Returns
    -------
    aligned_train : pd.DataFrame
        Salinan ``train_df`` bertipe kategori.
    aligned_others : list[pd.DataFrame]
        Salinan masing-masing DataFrame lain dengan ``CategoricalDtype`` identik.
    """
    tr = train_df.copy()
    others = [d.copy() for d in other_dfs]
    for col in cat_cols:
        if col not in tr.columns:
            continue
        cats = sorted(tr[col].dropna().astype(str).unique().tolist())
        cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
        tr[col] = tr[col].astype(str).astype(cat_type)
        for o in others:
            if col in o.columns:
                o[col] = o[col].astype(str).astype(cat_type)
    return tr, others
