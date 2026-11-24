"""Pemodelan dan Tuning di Dalam Validasi Waktu untuk Tugas 1c (Regresi).

Membandingkan satu baseline sederhana dan empat keluarga model lanjutan yang
seluruhnya dituning ketat di dalam skema Expanding-Window Time-Based CV
(3 fold x 2 bulan dari ``validation.py``):
0. ``BaselineGroupQuantile``   : P50 & P90 empiris per ``(layanan, pulau_tujuan)``.
1. ``LogTargetRidgeQuantile``  : Model linear Ridge pada target ``log(durasi_jam)``
   dengan kuantil residual heteroskedastik per ``layanan``.
2. ``LightGBM Quantile``       : Gradient boosting leaf-wise (uji raw vs log-target,
   dengan vs tanpa ID ``mitra_kurir``, dan tuning hiperparameter).
3. ``CatBoost MultiQuantile``  : Symmetric oblivious trees dengan loss
   ``MultiQuantile:alpha=0.5,0.9`` dan native ordered categories.
4. ``XGBoost MultiQuantile``   : Histogram depth-wise trees dengan
   ``reg:quantileerror`` bersama untuk kuantil ``[0.5, 0.9]``.
5. ``Ensemble Blend``          : Kombinasi konveks model-model terbaik pada P50 & P90.

Jalankan langsung:
    python kode/src/regresi/models.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    N_JOBS,
    REGRESI_TRAIN_PATH,
    RESULTS_DIR,
    SEED,
    siapkan_direktori,
)
from src.regresi.features import Task1FeaturePipeline  # noqa: E402
from src.regresi.preprocess import CATEGORICAL_COLUMNS, TARGET_COLUMN, perbaiki_satuan_target  # noqa: E402
from src.regresi.validation import TimeFold, buat_fold_waktu  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    atur_gaya_grafik,
    label_batang,
    miring,
    hitung_cakupan_p90,
    hitung_pinball_loss,
    tulis_json,
    atur_seed_global,
    sumbu_koma,
    catat_waktu,
)

LOW_CARD_CATS = ("layanan", "hub_asal", "pulau_tujuan")
CATBOOST_CAT_COLS = ("layanan", "hub_asal", "kota_asal", "pulau_tujuan", "kota_tujuan")


@dataclass
class PreparedFold:
    """Satu fold validasi waktu yang matriks fiturnya telah disiapkan anti-leakage."""

    fold: TimeFold
    tr_raw: pd.DataFrame
    va_raw: pd.DataFrame
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray


def siapkan_semua_fold(train: pd.DataFrame) -> list[PreparedFold]:
    """Siapkan matriks fitur anti-leakage sekali per fold untuk dipakai semua model.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.

    Returns
    -------
    list[PreparedFold]
        Daftar 3 fold kronologis siap latih dan evaluasi.
    """
    folds = buat_fold_waktu(train)
    prepared: list[PreparedFold] = []
    for f in folds:
        tr_raw = train.iloc[f.train_idx].reset_index(drop=True)
        va_raw = train.iloc[f.val_idx].reset_index(drop=True)
        y_tr, _ = perbaiki_satuan_target(tr_raw[TARGET_COLUMN])
        y_va, _ = perbaiki_satuan_target(va_raw[TARGET_COLUMN])
        pipe = Task1FeaturePipeline(seed=SEED)
        X_tr = pipe.fit_transform(tr_raw, y_tr)
        X_va = pipe.transform(va_raw)
        prepared.append(
            PreparedFold(
                fold=f,
                tr_raw=tr_raw,
                va_raw=va_raw,
                X_tr=X_tr,
                X_va=X_va,
                y_tr=y_tr,
                y_va=y_va,
            )
        )
    return prepared


def _eval_fold_preds(y_true: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict[str, float]:
    """Hitung metrik standar satu fold (Pinball P50, Pinball P90, Coverage, MAE)."""
    p90_safe = np.maximum(p50, p90)
    return {
        "pinball_p50": round(hitung_pinball_loss(y_true, p50, 0.5), 4),
        "pinball_p90": round(hitung_pinball_loss(y_true, p90_safe, 0.9), 4),
        "p90_coverage": round(hitung_cakupan_p90(y_true, p90_safe), 4),
        "mae_p50": round(float(mean_absolute_error(y_true, p50)), 4),
    }


def _ringkas_cv(per_fold: list[dict[str, Any]]) -> dict[str, float]:
    """Hitung rata-rata dan standar deviasi lintas 3 fold Time CV + skor Fold 3."""
    out: dict[str, float] = {}
    for k in ("pinball_p50", "pinball_p90", "p90_coverage", "mae_p50"):
        vals = [f[k] for f in per_fold]
        out[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        out[f"{k}_std"] = round(float(np.std(vals, ddof=0)), 4)
        out[f"fold3_{k}"] = per_fold[-1][k]
    return out


# ---------------------------------------------------------------------------
# Model 0: Baseline sederhana — kuantil empiris per (layanan, pulau_tujuan)
# ---------------------------------------------------------------------------
@dataclass
class BaselineGroupQuantile:
    """Baseline sederhana: median (P50) dan kuantil 90% (P90) per ``(layanan, pulau_tujuan)``."""

    group_q50_: dict[tuple[str, str], float] = field(default_factory=dict)
    group_q90_: dict[tuple[str, str], float] = field(default_factory=dict)
    service_q50_: dict[str, float] = field(default_factory=dict)
    service_q90_: dict[str, float] = field(default_factory=dict)
    global_q50_: float = 60.0
    global_q90_: float = 120.0

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> BaselineGroupQuantile:
        """Pelajari kuantil empiris P50 dan P90 dari data latih."""
        tmp = pd.DataFrame(
            {
                "layanan": df["layanan"].astype(str).to_numpy(),
                "pulau": df["pulau_tujuan"].astype(str).to_numpy(),
                "y": y,
            }
        )
        self.global_q50_ = float(np.quantile(y, 0.50))
        self.global_q90_ = float(np.quantile(y, 0.90))
        self.service_q50_ = tmp.groupby("layanan")["y"].quantile(0.50).to_dict()
        self.service_q90_ = tmp.groupby("layanan")["y"].quantile(0.90).to_dict()
        self.group_q50_ = tmp.groupby(["layanan", "pulau"])["y"].quantile(0.50).to_dict()
        self.group_q90_ = tmp.groupby(["layanan", "pulau"])["y"].quantile(0.90).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prediksi P50 & P90 menggunakan lookup grup dengan fallback hierarkis."""
        lay = df["layanan"].astype(str).to_numpy()
        pul = df["pulau_tujuan"].astype(str).to_numpy()
        p50 = np.empty(len(df), dtype=np.float64)
        p90 = np.empty(len(df), dtype=np.float64)
        for i in range(len(df)):
            key = (lay[i], pul[i])
            p50[i] = self.group_q50_.get(key, self.service_q50_.get(lay[i], self.global_q50_))
            p90[i] = self.group_q90_.get(key, self.service_q90_.get(lay[i], self.global_q90_))
        return p50, np.maximum(p50, p90)


# ---------------------------------------------------------------------------
# Model 1: Model Linear Ridge pada log(durasi_jam) + residual kuantil per layanan
# ---------------------------------------------------------------------------
@dataclass
class LogTargetRidgeQuantile:
    """Model linear Ridge dengan target transformasi ``log(durasi_jam)``.

    Karena durasi pengiriman bersifat multiplikatif terhadap jarak, kecepatan
    kurir (``f35``), dan jenis layanan, transformasi ``log(y)`` mengubah relasi
    multiplikatif menjadi aditif linear. Kuantil P50 dan P90 diperoleh dari
    kuantil residual ``log(y) - pred_log`` yang dihitung terpisah per ``layanan``
    (menangkap perbedaan varians antar layanan).
    """

    alpha: float = 100.0
    seed: int = SEED
    pipeline_: Pipeline = field(init=False)
    num_cols_: list[str] = field(default_factory=list, init=False)
    res_q50_: dict[str, float] = field(default_factory=dict, init=False)
    res_q90_: dict[str, float] = field(default_factory=dict, init=False)
    global_res_q50_: float = 0.0
    global_res_q90_: float = 0.3

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogTargetRidgeQuantile:
        """Fit Ridge pada ``log(y)`` dan pelajari offset kuantil residual per layanan."""
        self.num_cols_ = [c for c in X.columns if c not in CATEGORICAL_COLUMNS]
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), self.num_cols_),
                ("cat", OneHotEncoder(handle_unknown="ignore"), list(LOW_CARD_CATS)),
            ]
        )
        self.pipeline_ = Pipeline([("pre", pre), ("reg", Ridge(alpha=self.alpha, random_state=self.seed))])
        log_y = np.log(np.maximum(y, 1e-3))
        self.pipeline_.fit(X[self.num_cols_ + list(LOW_CARD_CATS)], log_y)

        pred_log = self.pipeline_.predict(X[self.num_cols_ + list(LOW_CARD_CATS)])
        resid = log_y - pred_log
        self.global_res_q50_ = float(np.quantile(resid, 0.50))
        self.global_res_q90_ = float(np.quantile(resid, 0.90))
        tmp = pd.DataFrame({"layanan": X["layanan"].astype(str).to_numpy(), "r": resid})
        self.res_q50_ = tmp.groupby("layanan")["r"].quantile(0.50).to_dict()
        self.res_q90_ = tmp.groupby("layanan")["r"].quantile(0.90).to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Kembalikan prediksi P50 dan P90 pada skala jam asli via ``exp()``."""
        pred_log = self.pipeline_.predict(X[self.num_cols_ + list(LOW_CARD_CATS)])
        lay = X["layanan"].astype(str)
        off50 = lay.map(self.res_q50_).fillna(self.global_res_q50_).to_numpy(dtype=np.float64)
        off90 = lay.map(self.res_q90_).fillna(self.global_res_q90_).to_numpy(dtype=np.float64)
        p50 = np.exp(np.clip(pred_log + off50, 0.0, 8.0))
        p90 = np.exp(np.clip(pred_log + off90, 0.0, 8.0))
        return p50, np.maximum(p50, p90)


# ---------------------------------------------------------------------------
# Helper latih & prediksi untuk GBDT (LightGBM, CatBoost, XGBoost)
# ---------------------------------------------------------------------------
def _select_model_cols(X: pd.DataFrame, drop_mitra_id: bool) -> list[str]:
    """Pilih kolom fitur untuk model pohon (opsional buang ID ``mitra_kurir``)."""
    if drop_mitra_id:
        return [c for c in X.columns if c != "mitra_kurir"]
    return list(X.columns)


def latih_prediksi_lgbm(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    params: dict[str, Any],
    use_log_target: bool = True,
    drop_mitra_id: bool = True,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, lgb.LGBMRegressor, lgb.LGBMRegressor]:
    """Latih pasangan LightGBM Quantile (P50 & P90) dan kembalikan prediksi + model."""
    cols = _select_model_cols(X_tr, drop_mitra_id=drop_mitra_id)
    target = np.log(np.maximum(y_tr, 1e-3)) if use_log_target else y_tr

    base = {
        "random_state": seed,
        "n_jobs": N_JOBS,
        "verbose": -1,
        **params,
    }
    m50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **base).fit(X_tr[cols], target)
    m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **base).fit(X_tr[cols], target)

    raw50 = m50.predict(X_va[cols])
    raw90 = m90.predict(X_va[cols])
    if use_log_target:
        p50 = np.exp(np.clip(raw50, 0.0, 8.0))
        p90 = np.exp(np.clip(raw90, 0.0, 8.0))
    else:
        p50, p90 = raw50, raw90
    return p50, np.maximum(p50, p90), m50, m90


def _prep_catboost_df(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Konversi kolom kategorikal menjadi string aman (``'__UNSEEN__'`` untuk ``NaN``)."""
    out = df[cols].copy()
    cat_cols = [c for c in cols if c in CATEGORICAL_COLUMNS]
    for c in cat_cols:
        out[c] = out[c].astype(object).where(out[c].notna(), "__UNSEEN__").astype(str)
    return out, cat_cols


def latih_prediksi_catboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    params: dict[str, Any],
    use_log_target: bool = True,
    drop_mitra_id: bool = True,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, CatBoostRegressor]:
    """Latih CatBoost ``MultiQuantile:alpha=0.5,0.9`` dan kembalikan prediksi + model."""
    cols = _select_model_cols(X_tr, drop_mitra_id=drop_mitra_id)
    tr_cb, cat_cols = _prep_catboost_df(X_tr, cols)
    va_cb, _ = _prep_catboost_df(X_va, cols)
    target = np.log(np.maximum(y_tr, 1e-3)) if use_log_target else y_tr

    model = CatBoostRegressor(
        loss_function="MultiQuantile:alpha=0.5,0.9",
        random_seed=seed,
        thread_count=N_JOBS,
        verbose=0,
        **params,
    )
    model.fit(tr_cb, target, cat_features=cat_cols)
    preds = model.predict(va_cb)
    if use_log_target:
        p50 = np.exp(np.clip(preds[:, 0], 0.0, 8.0))
        p90 = np.exp(np.clip(preds[:, 1], 0.0, 8.0))
    else:
        p50, p90 = preds[:, 0], preds[:, 1]
    return p50, np.maximum(p50, p90), model


def latih_prediksi_xgboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    params: dict[str, Any],
    use_log_target: bool = True,
    drop_mitra_id: bool = True,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, xgb.XGBRegressor]:
    """Latih XGBoost ``reg:quantileerror`` bersama untuk ``[0.5, 0.9]``."""
    cols = _select_model_cols(X_tr, drop_mitra_id=drop_mitra_id)
    target = np.log(np.maximum(y_tr, 1e-3)) if use_log_target else y_tr

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=np.array([0.5, 0.9]),
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        **params,
    )
    model.fit(X_tr[cols], target)
    preds = model.predict(X_va[cols])
    if use_log_target:
        p50 = np.exp(np.clip(preds[:, 0], 0.0, 8.0))
        p90 = np.exp(np.clip(preds[:, 1], 0.0, 8.0))
    else:
        p50, p90 = preds[:, 0], preds[:, 1]
    return p50, np.maximum(p50, p90), model


def optimasi_bobot_ensemble(
    preds_list: list[np.ndarray],
    y_true: np.ndarray,
    q: float,
) -> np.ndarray:
    """Cari bobot konveks (``w >= 0``, ``sum(w) = 1``) peminimum Pinball Loss kuantil ``q``."""
    P = np.column_stack(preds_list)
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k, dtype=np.float64)

    def obj(w: np.ndarray) -> float:
        return hitung_pinball_loss(y_true, P @ w, q)

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


def jalankan_perbandingan_dan_tuning(train: pd.DataFrame) -> dict[str, Any]:
    """Jalankan benchmark baseline + tuning 4 keluarga model + ensemble di dalam Time CV.

    Seluruh keputusan tuning diambil ketat berdasarkan skor rata-rata 3-fold
    Expanding-Window Time-Based CV tanpa pernah menyentuh data uji.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.

    Returns
    -------
    dict[str, Any]
        Hasil lengkap tuning tiap model, perbandingan akhir, bobot ensemble,
        serta menyimpan prediksi OOF ke ``kode/cache/task1_oof.npz``.
    """
    siapkan_direktori()
    with catat_waktu("Menyiapkan 3 fold Time CV dengan feature pipeline anti-leakage"):
        folds = siapkan_semua_fold(train)

    tuning_log: dict[str, list[dict[str, Any]]] = {}
    best_models_oof: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    summary_table: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 0. Baseline sederhana
    # ------------------------------------------------------------------
    with catat_waktu("Model 0: Baseline Group Quantile (layanan x pulau_tujuan)"):
        base_folds, base_oof = [], []
        for pf in folds:
            m = BaselineGroupQuantile().fit(pf.tr_raw, pf.y_tr)
            p50, p90 = m.predict(pf.va_raw)
            base_oof.append((p50, p90))
            m_res = _eval_fold_preds(pf.y_va, p50, p90)
            m_res["fold"] = pf.fold.name
            base_folds.append(m_res)
        best_models_oof["baseline"] = base_oof
        summary_table["baseline"] = {"per_fold": base_folds, "ringkasan": _ringkas_cv(base_folds)}

    # ------------------------------------------------------------------
    # 1. Model Linear: Log-Target Ridge (tuning alpha di dalam Time CV)
    # ------------------------------------------------------------------
    with catat_waktu("Model 1: Log-Target Ridge Quantile (tuning alpha)"):
        ridge_grid = [1.0, 10.0, 100.0, 500.0]
        ridge_trials: list[dict[str, Any]] = []
        best_ridge_score = float("inf")
        best_ridge_cfg: dict[str, Any] = {}
        best_ridge_oof: list[tuple[np.ndarray, np.ndarray]] = []
        best_ridge_folds: list[dict[str, Any]] = []

        for alpha in ridge_grid:
            f_res, f_oof = [], []
            for pf in folds:
                m = LogTargetRidgeQuantile(alpha=alpha, seed=SEED).fit(pf.X_tr, pf.y_tr)
                p50, p90 = m.predict(pf.X_va)
                f_oof.append((p50, p90))
                r = _eval_fold_preds(pf.y_va, p50, p90)
                r["fold"] = pf.fold.name
                f_res.append(r)
            ring = _ringkas_cv(f_res)
            combo = ring["pinball_p50_mean"] + ring["pinball_p90_mean"]
            ridge_trials.append({"alpha": alpha, "combined_pinball": round(combo, 4), **ring})
            if combo < best_ridge_score:
                best_ridge_score = combo
                best_ridge_cfg = {"alpha": alpha}
                best_ridge_oof = f_oof
                best_ridge_folds = f_res

        tuning_log["log_ridge"] = ridge_trials
        best_models_oof["log_ridge"] = best_ridge_oof
        summary_table["log_ridge"] = {
            "best_config": best_ridge_cfg,
            "per_fold": best_ridge_folds,
            "ringkasan": _ringkas_cv(best_ridge_folds),
        }

    # ------------------------------------------------------------------
    # 2. LightGBM Quantile (tuning target transform, drop_mitra_id, & hyperparams)
    # ------------------------------------------------------------------
    with catat_waktu("Model 2: LightGBM Quantile (tuning di dalam 3-fold Time CV)"):
        lgb_configs: list[dict[str, Any]] = [
            {
                "label": "raw_with_mitra_d63",
                "use_log_target": False,
                "drop_mitra_id": False,
                "params": {"n_estimators": 500, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 2.0},
            },
            {
                "label": "log_with_mitra_d63",
                "use_log_target": True,
                "drop_mitra_id": False,
                "params": {"n_estimators": 500, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 2.0},
            },
            {
                "label": "log_no_mitra_d63",
                "use_log_target": True,
                "drop_mitra_id": True,
                "params": {"n_estimators": 600, "learning_rate": 0.035, "num_leaves": 63, "min_child_samples": 60, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.75, "reg_lambda": 5.0},
            },
            {
                "label": "log_no_mitra_d31_reg",
                "use_log_target": True,
                "drop_mitra_id": True,
                "params": {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 80, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.75, "reg_lambda": 10.0},
            },
        ]
        lgb_trials: list[dict[str, Any]] = []
        best_lgb_score = float("inf")
        best_lgb_cfg: dict[str, Any] = {}
        best_lgb_oof: list[tuple[np.ndarray, np.ndarray]] = []
        best_lgb_folds: list[dict[str, Any]] = []

        for cfg in lgb_configs:
            f_res, f_oof = [], []
            for pf in folds:
                p50, p90, _, _ = latih_prediksi_lgbm(
                    pf.X_tr,
                    pf.y_tr,
                    pf.X_va,
                    params=cfg["params"],
                    use_log_target=cfg["use_log_target"],
                    drop_mitra_id=cfg["drop_mitra_id"],
                    seed=SEED,
                )
                f_oof.append((p50, p90))
                r = _eval_fold_preds(pf.y_va, p50, p90)
                r["fold"] = pf.fold.name
                f_res.append(r)
            ring = _ringkas_cv(f_res)
            combo = ring["pinball_p50_mean"] + ring["pinball_p90_mean"]
            lgb_trials.append({"label": cfg["label"], "combined_pinball": round(combo, 4), **ring})
            if combo < best_lgb_score:
                best_lgb_score = combo
                best_lgb_cfg = cfg
                best_lgb_oof = f_oof
                best_lgb_folds = f_res

        tuning_log["lgbm"] = lgb_trials
        best_models_oof["lgbm"] = best_lgb_oof
        summary_table["lgbm"] = {
            "best_config": best_lgb_cfg,
            "per_fold": best_lgb_folds,
            "ringkasan": _ringkas_cv(best_lgb_folds),
        }

    # ------------------------------------------------------------------
    # 3. CatBoost MultiQuantile (tuning depth & regularization)
    # ------------------------------------------------------------------
    with catat_waktu("Model 3: CatBoost MultiQuantile (tuning di dalam 3-fold Time CV)"):
        cb_configs: list[dict[str, Any]] = [
            {
                "label": "cb_log_d6_l2_5",
                "use_log_target": True,
                "drop_mitra_id": True,
                "params": {"iterations": 900, "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 5.0},
            },
            {
                "label": "cb_log_d8_l2_10",
                "use_log_target": True,
                "drop_mitra_id": True,
                "params": {"iterations": 800, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 10.0},
            },
        ]
        cb_trials: list[dict[str, Any]] = []
        best_cb_score = float("inf")
        best_cb_cfg: dict[str, Any] = {}
        best_cb_oof: list[tuple[np.ndarray, np.ndarray]] = []
        best_cb_folds: list[dict[str, Any]] = []

        for cfg in cb_configs:
            f_res, f_oof = [], []
            for pf in folds:
                p50, p90, _ = latih_prediksi_catboost(
                    pf.X_tr,
                    pf.y_tr,
                    pf.X_va,
                    params=cfg["params"],
                    use_log_target=cfg["use_log_target"],
                    drop_mitra_id=cfg["drop_mitra_id"],
                    seed=SEED,
                )
                f_oof.append((p50, p90))
                r = _eval_fold_preds(pf.y_va, p50, p90)
                r["fold"] = pf.fold.name
                f_res.append(r)
            ring = _ringkas_cv(f_res)
            combo = ring["pinball_p50_mean"] + ring["pinball_p90_mean"]
            cb_trials.append({"label": cfg["label"], "combined_pinball": round(combo, 4), **ring})
            if combo < best_cb_score:
                best_cb_score = combo
                best_cb_cfg = cfg
                best_cb_oof = f_oof
                best_cb_folds = f_res

        tuning_log["catboost"] = cb_trials
        best_models_oof["catboost"] = best_cb_oof
        summary_table["catboost"] = {
            "best_config": best_cb_cfg,
            "per_fold": best_cb_folds,
            "ringkasan": _ringkas_cv(best_cb_folds),
        }

    # ------------------------------------------------------------------
    # 4. XGBoost MultiQuantile (tuning max_depth & min_child_weight)
    # ------------------------------------------------------------------
    with catat_waktu("Model 4: XGBoost MultiQuantile (tuning di dalam 3-fold Time CV)"):
        xgb_configs: list[dict[str, Any]] = [
            {
                "label": "xgb_log_d6_mcw50",
                "use_log_target": True,
                "drop_mitra_id": True,
                "params": {"n_estimators": 550, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.75, "reg_lambda": 5.0},
            },
            {
                "label": "xgb_log_d8_mcw80",
                "use_log_target": True,
                "drop_mitra_id": True,
                "params": {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 8, "min_child_weight": 80, "subsample": 0.8, "colsample_bytree": 0.75, "reg_lambda": 10.0},
            },
        ]
        xgb_trials: list[dict[str, Any]] = []
        best_xgb_score = float("inf")
        best_xgb_cfg: dict[str, Any] = {}
        best_xgb_oof: list[tuple[np.ndarray, np.ndarray]] = []
        best_xgb_folds: list[dict[str, Any]] = []

        for cfg in xgb_configs:
            f_res, f_oof = [], []
            for pf in folds:
                p50, p90, _ = latih_prediksi_xgboost(
                    pf.X_tr,
                    pf.y_tr,
                    pf.X_va,
                    params=cfg["params"],
                    use_log_target=cfg["use_log_target"],
                    drop_mitra_id=cfg["drop_mitra_id"],
                    seed=SEED,
                )
                f_oof.append((p50, p90))
                r = _eval_fold_preds(pf.y_va, p50, p90)
                r["fold"] = pf.fold.name
                f_res.append(r)
            ring = _ringkas_cv(f_res)
            combo = ring["pinball_p50_mean"] + ring["pinball_p90_mean"]
            xgb_trials.append({"label": cfg["label"], "combined_pinball": round(combo, 4), **ring})
            if combo < best_xgb_score:
                best_xgb_score = combo
                best_xgb_cfg = cfg
                best_xgb_oof = f_oof
                best_xgb_folds = f_res

        tuning_log["xgboost"] = xgb_trials
        best_models_oof["xgboost"] = best_xgb_oof
        summary_table["xgboost"] = {
            "best_config": best_xgb_cfg,
            "per_fold": best_xgb_folds,
            "ringkasan": _ringkas_cv(best_xgb_folds),
        }

    # ------------------------------------------------------------------
    # 5. Ensemble Blend (bobot dipelajari jujur pada Fold 1+2, diuji di Fold 3;
    #    juga hitung bobot final dari seluruh 3 fold OOF untuk prediksi test)
    # ------------------------------------------------------------------
    members = ["log_ridge", "lgbm", "catboost", "xgboost"]
    y_f12 = np.concatenate([folds[0].y_va, folds[1].y_va])
    p50_f12 = [np.concatenate([best_models_oof[m][0][0], best_models_oof[m][1][0]]) for m in members]
    p90_f12 = [np.concatenate([best_models_oof[m][0][1], best_models_oof[m][1][1]]) for m in members]
    w50_f12 = optimasi_bobot_ensemble(p50_f12, y_f12, q=0.5)
    w90_f12 = optimasi_bobot_ensemble(p90_f12, y_f12, q=0.9)

    # Bobot final pada semua 3 fold OOF (untuk produksi Sep-Okt)
    y_all = np.concatenate([pf.y_va for pf in folds])
    p50_all = [np.concatenate([best_models_oof[m][i][0] for i in range(3)]) for m in members]
    p90_all = [np.concatenate([best_models_oof[m][i][1] for i in range(3)]) for m in members]
    w50_final = optimasi_bobot_ensemble(p50_all, y_all, q=0.5)
    w90_final = optimasi_bobot_ensemble(p90_all, y_all, q=0.9)

    ens_folds, ens_oof = [], []
    for i, pf in enumerate(folds):
        # Pada Fold 3 gunakan bobot dari Fold 1+2 agar evaluasi Fold 3 100% out-of-sample!
        w50 = w50_f12 if i == 2 else w50_final
        w90 = w90_f12 if i == 2 else w90_final
        p50 = sum(w50[j] * best_models_oof[m][i][0] for j, m in enumerate(members))
        p90 = np.maximum(p50, sum(w90[j] * best_models_oof[m][i][1] for j, m in enumerate(members)))
        ens_oof.append((p50, p90))
        r = _eval_fold_preds(pf.y_va, p50, p90)
        r["fold"] = pf.fold.name
        ens_folds.append(r)

    best_models_oof["ensemble"] = ens_oof
    summary_table["ensemble"] = {
        "members": members,
        "weights_p50_fold12_honest": {m: round(float(w50_f12[j]), 4) for j, m in enumerate(members)},
        "weights_p90_fold12_honest": {m: round(float(w90_f12[j]), 4) for j, m in enumerate(members)},
        "weights_p50_final_all3": {m: round(float(w50_final[j]), 4) for j, m in enumerate(members)},
        "weights_p90_final_all3": {m: round(float(w90_final[j]), 4) for j, m in enumerate(members)},
        "per_fold": ens_folds,
        "ringkasan": _ringkas_cv(ens_folds),
    }

    # Simpan OOF Fold 2 & Fold 3 ke cache untuk Step 1e (Conformal Calibration) & Step 1f (Analisis)
    cache_payload: dict[str, np.ndarray] = {}
    for i, pf in enumerate(folds):
        cache_payload[f"y_fold{i+1}"] = pf.y_va
        cache_payload[f"idx_fold{i+1}"] = pf.fold.val_idx
        for m in ("baseline", "log_ridge", "lgbm", "catboost", "xgboost", "ensemble"):
            cache_payload[f"{m}_p50_fold{i+1}"] = best_models_oof[m][i][0]
            cache_payload[f"{m}_p90_fold{i+1}"] = best_models_oof[m][i][1]
    np.savez_compressed(CACHE_DIR / "task1_oof.npz", **cache_payload)

    # Buat grafik perbandingan model (Tugas 1c)
    fig_path = plot_perbandingan_model(summary_table)

    result = {
        "task": "Tugas 1c — Perbandingan Baseline, 4 Model Lanjutan, Tuning Time-CV, & Ensemble",
        "seed": SEED,
        "models": summary_table,
        "tuning_log": tuning_log,
        "figure": fig_path.name,
    }
    tulis_json(result, RESULTS_DIR / "task1_models.json")
    return result


def plot_perbandingan_model(summary_table: dict[str, Any]) -> Path:
    """Buat grafik 2-panel perbandingan pinball loss P50 dan P90 semua model pada validasi berbasis waktu."""
    siapkan_direktori()
    atur_gaya_grafik()

    order = [
        ("baseline", "Model\ndasar"),
        ("log_ridge", "Ridge"),
        ("lgbm", "LightGBM"),
        ("catboost", "CatBoost"),
        ("xgboost", "XGBoost"),
        ("ensemble", "Model\ngabungan"),
    ]
    labels = [lbl for _, lbl in order]
    p50_mean = [summary_table[k]["ringkasan"]["pinball_p50_mean"] for k, _ in order]
    p90_mean = [summary_table[k]["ringkasan"]["pinball_p90_mean"] for k, _ in order]
    p50_f3 = [summary_table[k]["ringkasan"]["fold3_pinball_p50"] for k, _ in order]
    p90_f3 = [summary_table[k]["ringkasan"]["fold3_pinball_p90"] for k, _ in order]

    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.7))
    x = np.arange(len(labels))
    w = 0.38

    for ax, v50, v90, title in [
        (axes[0], p50_mean, p90_mean, f"(a) Rata-rata Tiga {miring('Fold')}"),
        (axes[1], p50_f3, p90_f3, f"(b) {miring('Fold')} 3 (Validasi Juli s.d. Agustus 2025)"),
    ]:
        b1 = ax.bar(x - w / 2, v50, width=w, color="#2563eb", label="P50")
        b2 = ax.bar(x + w / 2, v90, width=w, color="#f97316", label="P90")
        label_batang(ax, b1, putar=True)
        label_batang(ax, b2, putar=True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel(f"{miring('Pinball loss')} (jam)")
        ax.set_title(title)
        ax.set_ylim(0, max(v50 + v90) * 1.22)
        ax.grid(axis="x", visible=False)
        sumbu_koma(ax, "y")
        ax.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1c_model_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    """Jalankan benchmark & tuning Tugas 1c dari CLI."""
    atur_seed_global(SEED)
    train = pd.read_csv(REGRESI_TRAIN_PATH)
    res = jalankan_perbandingan_dan_tuning(train)

    print("\n================ RINGKASAN MODEL TUGAS 1c (TIME CV) ================")
    for k in ("baseline", "log_ridge", "lgbm", "catboost", "xgboost", "ensemble"):
        r = res["models"][k]["ringkasan"]
        print(
            f"  {k:12s} | Mean3F: P50={r['pinball_p50_mean']:.4f}, P90={r['pinball_p90_mean']:.4f}, cov={r['p90_coverage_mean']*100:.2f}% "
            f"| Fold3(Jul-Agu): P50={r['fold3_pinball_p50']:.4f}, P90={r['fold3_pinball_p90']:.4f}, cov={r['fold3_p90_coverage']*100:.2f}%"
        )
    print("\nBobot Ensemble Final (3-fold OOF):")
    print("  P50:", res["models"]["ensemble"]["weights_p50_final_all3"])
    print("  P90:", res["models"]["ensemble"]["weights_p90_final_all3"])
    print(f"\nGrafik: kode/figures/{res['figure']}")
    print("JSON  : kode/results/task1_models.json")


if __name__ == "__main__":
    main()
