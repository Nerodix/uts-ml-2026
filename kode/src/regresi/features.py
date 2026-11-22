"""Rekayasa fitur, penanganan kategori high-cardinality, & uji kategori unseen (Tugas 1d).

Modul ini membangun fitur berbasis waktu pickup (100% bebas leakage masa depan)
dan menangani dua tantangan kategori utama Tugas 1d:
1. **High-cardinality (`kota_tujuan` 405 kota & rute `hub_asal × kota_tujuan`):**
   *Hierarchical Smoothed Target Encoding* (Out-of-Fold di train agar bebas
   target leakage, dengan *empirical Bayes shrinkage* bertingkat menuju
   ``pulau_tujuan`` dan rata-rata global).
2. **Kategori *unseen* di data uji (`mitra_kurir = K13`, 3,9% data uji):**
   Fallback kategori aman + pemanfaatan fitur sinyal ``f35`` (indeks kecepatan
   kurir real-time), dibuktikan lewat simulasi *Leave-One-Courier-Out*.

Jalankan simulasi unseen courier:
    python kode/src/regresi/features.py --uji-unseen
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    N_JOBS,
    REGRESI_TRAIN_PATH,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.regresi.preprocess import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    DATE_COLUMN,
    NAMED_NUMERIC_COLUMNS,
    SIGNAL_F_COLUMNS,
    TARGET_COLUMN,
    BeratImputer,
    RouteDistanceImputer,
    align_categorical_dtypes,
    fix_target_unit_errors,
)
from src.utils import (  # noqa: E402
    configure_plotting,
    p90_coverage,
    pinball_loss,
    save_json,
    set_global_seed,
    timer,
)

# Tanggal kembar e-commerce 2024–2025 (12.12, 1.1, 2.2, ..., 12.12)
TWIN_DATES_NS = np.array(
    [np.datetime64("2024-12-12", "D")]
    + [np.datetime64(f"2025-{m:02d}-{m:02d}", "D") for m in range(1, 13)],
    dtype="datetime64[D]",
)

JAWA_HUBS = frozenset({"JKT", "BDG", "SMG", "SBY"})


def tambah_fitur_kalender_dan_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur kalender pickup dan transformasi domain non-linear (tanpa target).

    Seluruh fitur dihitung murni dari informasi yang tersedia pada saat kurir
    mengambil paket (``tgl_pickup``, ``jam_pickup``, jarak, berat/volume, antrean,
    hujan, dan 10 fitur sinyal ``f``).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame regresi setelah ``RouteDistanceImputer`` dan ``BeratImputer``.

    Returns
    -------
    pd.DataFrame
        Salinan DataFrame dengan tambahan fitur kalender dan interaksi fisik.
    """
    out = df.copy()
    dt = pd.to_datetime(out[DATE_COLUMN])
    dt_days = dt.to_numpy(dtype="datetime64[D]")

    # 1. Fitur kalender & siklus jam pickup
    out["day_of_week"] = dt.dt.day_of_week.astype(np.int8)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(np.int8)
    out["day_of_month"] = dt.dt.day.astype(np.int8)
    out["is_payday_window"] = ((out["day_of_month"] >= 25) & (out["day_of_month"] <= 29)).astype(np.int8)
    jam_rad = 2.0 * np.pi * out["jam_pickup"].to_numpy(dtype=np.float64) / 24.0
    out["jam_sin"] = np.sin(jam_rad)
    out["jam_cos"] = np.cos(jam_rad)

    # 2. Jarak hari sejak tanggal kembar terdekat (0 = hari H, 1–2 = puncak antrean)
    idx = np.searchsorted(TWIN_DATES_NS, dt_days, side="right") - 1
    days_since_twin = (dt_days - TWIN_DATES_NS[np.clip(idx, 0, len(TWIN_DATES_NS) - 1)]).astype(np.int32)
    out["days_since_twin"] = np.clip(days_since_twin, 0, 7).astype(np.int8)
    out["is_twin_day0"] = (days_since_twin == 0).astype(np.int8)
    out["is_twin_day1_2"] = ((days_since_twin == 1) | (days_since_twin == 2)).astype(np.int8)
    out["is_twin_window"] = (days_since_twin <= 3).astype(np.int8)

    # 3. Indikator guncangan satu kali Lebaran 2025 (26 Mar - 6 Apr) agar tidak membiaskan bulan normal
    out["is_lebaran_2025"] = ((dt >= "2025-03-26") & (dt <= "2025-04-06")).astype(np.int8)

    # 4. Transformasi log & fitur fisik domain logistik
    out["log_jarak_km"] = np.log1p(np.maximum(out["jarak_km"], 0.0))
    out["log_berat_kg"] = np.log1p(np.maximum(out["berat_kg"], 0.0))
    out["log_volume_liter"] = np.log1p(np.maximum(out["volume_liter"], 0.0))
    out["log_nilai_barang"] = np.log1p(np.maximum(out["nilai_barang"], 0.0))
    out["log_antrean_hub"] = np.log1p(np.maximum(out["antrean_hub_asal"], 0.0))
    out["log_hujan_asal"] = np.log1p(np.maximum(out["hujan_asal_mm"], 0.0))
    out["log_hujan_tujuan"] = np.log1p(np.maximum(out["hujan_tujuan_mm"], 0.0))
    out["hujan_total_mm"] = out["hujan_asal_mm"] + out["hujan_tujuan_mm"]
    out["densitas_paket"] = out["berat_kg"] / np.maximum(out["volume_liter"], 0.05)

    # 5. Non-linearitas & interaksi kunci dari temuan audit
    out["f13_pos"] = np.maximum(0.0, out["f13"])  # ReLU kink yang ditemukan di audit
    out["log_f35"] = np.log(np.maximum(out["f35"], 0.1))
    out["jarak_x_f35"] = out["jarak_km"] * out["f35"]
    out["log_jarak_x_f35"] = out["log_jarak_km"] * out["f35"]
    out["antrean_x_f17"] = out["log_antrean_hub"] * out["f17"]
    out["jarak_x_alamat_risk"] = out["log_jarak_km"] * (1.0 - out["alamat_skor"])
    out["is_intra_jawa"] = (
        out["hub_asal"].isin(JAWA_HUBS) & (out["pulau_tujuan"] == "Jawa")
    ).astype(np.int8)

    # 6. Kunci rute komposit untuk target encoding hierarkis
    out["rute_hub_kota"] = out["hub_asal"].astype(str) + "__" + out["kota_tujuan"].astype(str)
    return out


@dataclass
class HierarchicalSmoothedTargetEncoder:
    """Out-of-Fold *Hierarchical Smoothed Target Encoder* untuk kategori high-cardinality.

    Catatan Anti-Leakage
    --------------------
    - Pada data latih (``fit_transform``), nilai encoding setiap baris dihitung
      secara *out-of-fold* (5-fold) sehingga baris tidak pernah memakai labelnya sendiri.
    - Pada data validasi/uji (``transform``), statistik diambil murni dari tabel
      data latih dengan *empirical Bayes shrinkage* (``m = 20``) menuju kategori
      induknya: ``rute_hub_kota -> kota_tujuan -> pulau_tujuan -> global``.
    - Target dienkode dalam skala ``log(durasi_jam)`` untuk rata-rata dan kuantil P90.
    """

    smoothing: float = 20.0
    n_oof_splits: int = 5
    seed: int = SEED
    global_mean_: float = 0.0
    global_p90_: float = 0.0
    pulau_mean_: dict[str, float] = field(default_factory=dict)
    pulau_p90_: dict[str, float] = field(default_factory=dict)
    kota_mean_: dict[str, float] = field(default_factory=dict)
    kota_p90_: dict[str, float] = field(default_factory=dict)
    kota_log_freq_: dict[str, float] = field(default_factory=dict)
    rute_mean_: dict[str, float] = field(default_factory=dict)

    def _fit_tables(self, df: pd.DataFrame, log_y: np.ndarray) -> dict[str, Any]:
        """Hitung tabel smoothed hierarkis pada satu subset latih."""
        m = self.smoothing
        tmp = pd.DataFrame(
            {
                "pulau": df["pulau_tujuan"].astype(str).to_numpy(),
                "kota": df["kota_tujuan"].astype(str).to_numpy(),
                "rute": df["rute_hub_kota"].astype(str).to_numpy(),
                "z": log_y,
            }
        )
        g_mean = float(np.mean(log_y))
        g_p90 = float(np.quantile(log_y, 0.90))

        # Level 1: pulau_tujuan (shrink ke global)
        p_grp = tmp.groupby("pulau")["z"].agg(
            n="size", mean="mean", p90=lambda s: float(np.quantile(s, 0.90))
        )
        pulau_mean = ((p_grp["n"] * p_grp["mean"] + m * g_mean) / (p_grp["n"] + m)).to_dict()
        pulau_p90 = ((p_grp["n"] * p_grp["p90"] + m * g_p90) / (p_grp["n"] + m)).to_dict()

        # Level 2: kota_tujuan (shrink ke pulau induknya)
        kota_to_pulau = tmp.drop_duplicates("kota").set_index("kota")["pulau"].to_dict()
        k_grp = tmp.groupby("kota")["z"].agg(
            n="size", mean="mean", p90=lambda s: float(np.quantile(s, 0.90))
        )
        k_prior_mean = k_grp.index.map(lambda k: pulau_mean.get(kota_to_pulau.get(k, ""), g_mean))
        k_prior_p90 = k_grp.index.map(lambda k: pulau_p90.get(kota_to_pulau.get(k, ""), g_p90))
        kota_mean = ((k_grp["n"] * k_grp["mean"] + m * k_prior_mean) / (k_grp["n"] + m)).to_dict()
        kota_p90 = ((k_grp["n"] * k_grp["p90"] + m * k_prior_p90) / (k_grp["n"] + m)).to_dict()
        kota_log_freq = np.log1p(k_grp["n"]).to_dict()

        # Level 3: rute_hub_kota (shrink ke kota_tujuan induknya)
        rute_to_kota = tmp.drop_duplicates("rute").set_index("rute")["kota"].to_dict()
        r_grp = tmp.groupby("rute")["z"].agg(n="size", mean="mean")
        r_prior_mean = r_grp.index.map(lambda r: kota_mean.get(rute_to_kota.get(r, ""), g_mean))
        rute_mean = ((r_grp["n"] * r_grp["mean"] + m * r_prior_mean) / (r_grp["n"] + m)).to_dict()

        return {
            "global_mean": g_mean,
            "global_p90": g_p90,
            "pulau_mean": pulau_mean,
            "pulau_p90": pulau_p90,
            "kota_mean": kota_mean,
            "kota_p90": kota_p90,
            "kota_log_freq": kota_log_freq,
            "rute_mean": rute_mean,
        }

    def _apply_tables(self, df: pd.DataFrame, tables: dict[str, Any]) -> pd.DataFrame:
        """Terapkan tabel lookup hierarkis ke DataFrame target."""
        pulau = df["pulau_tujuan"].astype(str)
        kota = df["kota_tujuan"].astype(str)
        rute = df["rute_hub_kota"].astype(str)

        p_mean = pulau.map(tables["pulau_mean"]).fillna(tables["global_mean"])
        p_p90 = pulau.map(tables["pulau_p90"]).fillna(tables["global_p90"])
        k_mean = kota.map(tables["kota_mean"]).fillna(p_mean)
        k_p90 = kota.map(tables["kota_p90"]).fillna(p_p90)
        r_mean = rute.map(tables["rute_mean"]).fillna(k_mean)
        k_freq = kota.map(tables["kota_log_freq"]).fillna(0.0)

        return pd.DataFrame(
            {
                "te_kota_log_mean": k_mean.to_numpy(dtype=np.float64),
                "te_kota_log_p90": k_p90.to_numpy(dtype=np.float64),
                "te_kota_log_spread": (k_p90 - k_mean).to_numpy(dtype=np.float64),
                "te_rute_log_mean": r_mean.to_numpy(dtype=np.float64),
                "kota_tujuan_log_freq": k_freq.to_numpy(dtype=np.float64),
            },
            index=df.index,
        )

    def fit_transform(self, train_df: pd.DataFrame, y_train_clean: np.ndarray) -> pd.DataFrame:
        """Pelajari tabel penuh di ``train_df`` dan kembalikan versi dengan fitur OOF TE.

        Parameters
        ----------
        train_df : pd.DataFrame
            Data latih setelah :func:`tambah_fitur_kalender_dan_domain`.
        y_train_clean : np.ndarray
            Target ``durasi_jam`` yang sudah dikoreksi satuan menit (``> 0``).

        Returns
        -------
        pd.DataFrame
            Salinan ``train_df`` dengan 5 kolom *out-of-fold target encoding*.
        """
        log_y = np.log(np.maximum(y_train_clean, 1e-3))
        kf = KFold(n_splits=self.n_oof_splits, shuffle=True, random_state=self.seed)

        oof_parts: list[pd.DataFrame] = []
        for tr_idx, va_idx in kf.split(train_df):
            sub_tables = self._fit_tables(train_df.iloc[tr_idx], log_y[tr_idx])
            oof_parts.append(self._apply_tables(train_df.iloc[va_idx], sub_tables))

        oof_df = pd.concat(oof_parts).reindex(train_df.index)

        # Simpan tabel penuh untuk transform() pada validasi / test
        full = self._fit_tables(train_df, log_y)
        self.global_mean_ = full["global_mean"]
        self.global_p90_ = full["global_p90"]
        self.pulau_mean_ = full["pulau_mean"]
        self.pulau_p90_ = full["pulau_p90"]
        self.kota_mean_ = full["kota_mean"]
        self.kota_p90_ = full["kota_p90"]
        self.kota_log_freq_ = full["kota_log_freq"]
        self.rute_mean_ = full["rute_mean"]

        out = train_df.copy()
        for col in oof_df.columns:
            out[col] = oof_df[col]
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Terapkan tabel target encoding dari ``fit_transform`` ke data validasi/uji.

        Parameters
        ----------
        df : pd.DataFrame
            Data validasi atau data uji setelah :func:`tambah_fitur_kalender_dan_domain`.

        Returns
        -------
        pd.DataFrame
            Salinan ``df`` dengan 5 kolom target encoding yang sama.
        """
        tables = {
            "global_mean": self.global_mean_,
            "global_p90": self.global_p90_,
            "pulau_mean": self.pulau_mean_,
            "pulau_p90": self.pulau_p90_,
            "kota_mean": self.kota_mean_,
            "kota_p90": self.kota_p90_,
            "kota_log_freq": self.kota_log_freq_,
            "rute_mean": self.rute_mean_,
        }
        te_df = self._apply_tables(df, tables)
        out = df.copy()
        for col in te_df.columns:
            out[col] = te_df[col]
        return out


@dataclass
class Task1FeaturePipeline:
    """Pipeline preprocessing + rekayasa fitur lengkap anti-leakage untuk Tugas 1.

    Menyatukan seluruh langkah persiapan secara aman:
    1. ``RouteDistanceImputer`` (fit hanya di train).
    2. ``BeratImputer`` (fit hanya di train).
    3. ``tambah_fitur_kalender_dan_domain`` (deterministik pickup-time).
    4. ``HierarchicalSmoothedTargetEncoder`` (OOF di train, lookup di val/test).
    5. Penyelarasan kategori (kategori *unseen* seperti ``K13`` menjadi ``NaN``).
    """

    smoothing: float = 20.0
    seed: int = SEED
    dist_imputer_: RouteDistanceImputer = field(init=False)
    berat_imputer_: BeratImputer = field(init=False)
    target_encoder_: HierarchicalSmoothedTargetEncoder = field(init=False)
    cat_dtypes_: dict[str, pd.CategoricalDtype] = field(default_factory=dict, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        y_train_clean: np.ndarray,
    ) -> pd.DataFrame:
        """Fit seluruh komponen pada ``train_df`` dan kembalikan matriks fitur siap latih.

        Parameters
        ----------
        train_df : pd.DataFrame
            Data latih mentah fold aktif.
        y_train_clean : np.ndarray
            Target ``durasi_jam`` terkoreksi (``/ 60`` untuk ``> 1000``).

        Returns
        -------
        pd.DataFrame
            DataFrame fitur bersih siap dipakai LightGBM/XGBoost/CatBoost/Linear.
        """
        self.dist_imputer_ = RouteDistanceImputer().fit(train_df)
        self.berat_imputer_ = BeratImputer().fit(train_df)
        self.target_encoder_ = HierarchicalSmoothedTargetEncoder(
            smoothing=self.smoothing, seed=self.seed
        )

        df = self.dist_imputer_.transform(train_df, add_indicator=True)
        df = self.berat_imputer_.transform(df, add_indicator=True)
        df = tambah_fitur_kalender_dan_domain(df)
        df = self.target_encoder_.fit_transform(df, y_train_clean)

        self.feature_columns_ = self._select_feature_columns(df)
        out = df[self.feature_columns_].copy()

        self.cat_dtypes_ = {}
        for col in CATEGORICAL_COLUMNS:
            cats = sorted(out[col].dropna().astype(str).unique().tolist())
            ct = pd.CategoricalDtype(categories=cats, ordered=False)
            self.cat_dtypes_[col] = ct
            out[col] = out[col].astype(str).astype(ct)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformasikan data validasi atau uji memakai parameter yang sudah di-fit.

        Parameters
        ----------
        df : pd.DataFrame
            Data validasi atau data uji mentah.

        Returns
        -------
        pd.DataFrame
            DataFrame fitur bersih dengan skema dan tipe kategori identik dengan train.
        """
        d = self.dist_imputer_.transform(df, add_indicator=True)
        d = self.berat_imputer_.transform(d, add_indicator=True)
        d = tambah_fitur_kalender_dan_domain(d)
        d = self.target_encoder_.transform(d)

        out = d[self.feature_columns_].copy()
        for col, ct in self.cat_dtypes_.items():
            out[col] = pd.Categorical(out[col].astype(str), categories=ct.categories, ordered=False)
        return out

    @staticmethod
    def _select_feature_columns(df: pd.DataFrame) -> list[str]:
        """Pilih daftar fitur bersih (bebas leakage, bebas 40 fitur noise f, bebas ID/tanggal)."""
        engineered = [
            "jarak_km_was_corrupt",
            "berat_kg_missing",
            "day_of_week",
            "is_weekend",
            "day_of_month",
            "is_payday_window",
            "jam_sin",
            "jam_cos",
            "days_since_twin",
            "is_twin_day0",
            "is_twin_day1_2",
            "is_twin_window",
            "is_lebaran_2025",
            "log_jarak_km",
            "log_berat_kg",
            "log_volume_liter",
            "log_nilai_barang",
            "log_antrean_hub",
            "log_hujan_asal",
            "log_hujan_tujuan",
            "hujan_total_mm",
            "densitas_paket",
            "f13_pos",
            "log_f35",
            "jarak_x_f35",
            "log_jarak_x_f35",
            "antrean_x_f17",
            "jarak_x_alamat_risk",
            "is_intra_jawa",
            "te_kota_log_mean",
            "te_kota_log_p90",
            "te_kota_log_spread",
            "te_rute_log_mean",
            "kota_tujuan_log_freq",
        ]
        cols = list(CATEGORICAL_COLUMNS) + list(NAMED_NUMERIC_COLUMNS) + list(SIGNAL_F_COLUMNS) + engineered
        return [c for c in cols if c in df.columns]


def uji_kategori_unseen_kurir(train: pd.DataFrame) -> dict[str, Any]:
    """Simulasi *Leave-One-Courier-Out* untuk membuktikan strategi menangani ``K13`` (Tugas 1d).

    Pada split Fold 3 (Train Jan–Jun -> Val Jul–Agu), tiga kurir dengan kecepatan
    berbeda (``K08`` cepat f35=0,855; ``K02`` lambat f35=1,080; ``K07`` bergeser
    f35=1,280) disimulasikan sebagai kurir *unseen* dengan menghapus barisnya dari
    train lalu mengevaluasi prediksi pada kurir tersebut di Val Jul–Agu:
    1. **Tanpa ``f35`` (hanya ID ``mitra_kurir``):** fallback ke rata-rata global.
    2. **Dengan ``mitra_kurir`` + ``f35``:** unseen jadi ``NaN``, ``f35`` menebak kecepatan.
    3. **Tanpa ID ``mitra_kurir`` (hanya ``f35``):** murni bergantung pada indeks kecepatan kurir.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.

    Returns
    -------
    dict[str, Any]
        Perbandingan Pinball P50, Pinball P90, dan bias prediksi per kurir simulasi unseen.
    """
    dt = pd.to_datetime(train[DATE_COLUMN])
    tr_full = train.loc[dt < "2025-07-01"].reset_index(drop=True)
    va_full = train.loc[dt >= "2025-07-01"].reset_index(drop=True)

    heldout_couriers = ["K08", "K02", "K07"]
    tr_sub = tr_full.loc[~tr_full["mitra_kurir"].isin(heldout_couriers)].reset_index(drop=True)
    va_unseen = va_full.loc[va_full["mitra_kurir"].isin(heldout_couriers)].reset_index(drop=True)

    y_tr, _ = fix_target_unit_errors(tr_sub[TARGET_COLUMN])
    y_va, _ = fix_target_unit_errors(va_unseen[TARGET_COLUMN])

    pipe = Task1FeaturePipeline(seed=SEED)
    X_tr_all = pipe.fit_transform(tr_sub, y_tr)
    X_va_all = pipe.transform(va_unseen)

    cols_no_f35 = [c for c in X_tr_all.columns if c not in ("f35", "log_f35", "jarak_x_f35", "log_jarak_x_f35")]
    cols_with_f35 = list(X_tr_all.columns)
    cols_f35_no_id = [c for c in X_tr_all.columns if c != "mitra_kurir"]

    params = {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }

    strategi: dict[str, dict[str, Any]] = {}
    for label, cols in [
        ("tanpa_f35_fallback_global", cols_no_f35),
        ("mitra_kurir_plus_f35", cols_with_f35),
        ("hanya_f35_tanpa_id_kurir", cols_f35_no_id),
    ]:
        m50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **params).fit(X_tr_all[cols], y_tr)
        m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **params).fit(X_tr_all[cols], y_tr)
        p50 = m50.predict(X_va_all[cols])
        p90 = np.maximum(p50, m90.predict(X_va_all[cols]))

        per_kurir: dict[str, dict[str, float]] = {}
        for k in heldout_couriers:
            mask = (va_unseen["mitra_kurir"] == k).to_numpy()
            per_kurir[k] = {
                "f35_median": round(float(va_unseen.loc[mask, "f35"].median()), 3),
                "pinball_p50": round(pinball_loss(y_va[mask], p50[mask], 0.5), 4),
                "pinball_p90": round(pinball_loss(y_va[mask], p90[mask], 0.9), 4),
                "bias_median": round(float(np.median(p50[mask] - y_va[mask])), 2),
                "p90_coverage": round(p90_coverage(y_va[mask], p90[mask]), 4),
            }

        strategi[label] = {
            "overall_pinball_p50": round(pinball_loss(y_va, p50, 0.5), 4),
            "overall_pinball_p90": round(pinball_loss(y_va, p90, 0.9), 4),
            "overall_p90_coverage": round(p90_coverage(y_va, p90), 4),
            "per_kurir_unseen": per_kurir,
        }

    # Plot bukti simulasi unseen courier
    ensure_dirs()
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    couriers = heldout_couriers
    x = np.arange(len(couriers))
    w = 0.26

    for idx_s, (key, title, col_hex) in enumerate(
        [
            ("tanpa_f35_fallback_global", "Tanpa f35 (fallback rata-rata)", "#dc2626"),
            ("mitra_kurir_plus_f35", "Kategori unseen->NaN + f35", "#2563eb"),
            ("hanya_f35_tanpa_id_kurir", "Murni f35 (tanpa ID kurir)", "#16a34a"),
        ]
    ):
        p50_k = [strategi[key]["per_kurir_unseen"][c]["pinball_p50"] for c in couriers]
        bias_k = [strategi[key]["per_kurir_unseen"][c]["bias_median"] for c in couriers]
        offset = (idx_s - 1) * w
        b1 = axes[0].bar(x + offset, p50_k, width=w, color=col_hex, label=title)
        axes[1].bar(x + offset, bias_k, width=w, color=col_hex, label=title)
        for bar in b1:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=7.2)

    lbl_c = ["K08\n(cepat, f35=0,86)", "K02\n(lambat, f35=1,08)", "K07\n(shift, f35=1,28)"]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lbl_c, fontsize=8.5)
    axes[0].set_ylabel("Pinball Loss P50 (jam)")
    axes[0].set_title("(a) Error pada Kurir Unseen (Simulasi K13)")
    axes[0].set_ylim(0, max(strategi["tanpa_f35_fallback_global"]["per_kurir_unseen"][c]["pinball_p50"] for c in couriers) * 1.2)
    axes[0].legend(fontsize=7.8)

    axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=1.1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lbl_c, fontsize=8.5)
    axes[1].set_ylabel("Bias Median Prediksi - Aktual (jam)")
    axes[1].set_title("(b) Penghilangan Bias pada Kurir Unseen lewat f35")
    axes[1].legend(fontsize=7.8)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "task1_1d_unseen_courier.png"
    fig.savefig(fig_path)
    plt.close(fig)

    out_json = RESULTS_DIR / "task1_unseen_courier.json"
    result = {
        "task": "Tugas 1d — Simulasi Kategori Unseen (Leave-One-Courier-Out)",
        "heldout_couriers": heldout_couriers,
        "n_val_unseen_rows": int(len(va_unseen)),
        "strategi": strategi,
        "figure": fig_path.name,
    }
    save_json(result, out_json)
    return result


def main() -> None:
    """Jalankan uji simulasi kategori unseen kurir dari CLI."""
    set_global_seed(SEED)
    parser = argparse.ArgumentParser(description="Feature engineering & uji unseen courier Tugas 1d.")
    parser.add_argument("--uji-unseen", action="store_true", help="Jalankan simulasi Leave-One-Courier-Out.")
    args = parser.parse_args()

    if args.uji_unseen:
        train = pd.read_csv(REGRESI_TRAIN_PATH)
        with timer("Simulasi Leave-One-Courier-Out (K08, K02, K07)"):
            res = uji_kategori_unseen_kurir(train)
        print("\n=== HASIL SIMULASI KURIR UNSEEN (TUGAS 1d) ===")
        for nama, s in res["strategi"].items():
            print(
                f"  {nama:28s} -> P50={s['overall_pinball_p50']:.4f} | "
                f"P90={s['overall_pinball_p90']:.4f} | cov90={s['overall_p90_coverage']*100:.2f}%"
            )
            for k, mk in s["per_kurir_unseen"].items():
                print(f"      {k} (f35={mk['f35_median']:.3f}): P50={mk['pinball_p50']:.3f}, bias={mk['bias_median']:+.2f} jam")


if __name__ == "__main__":
    main()
