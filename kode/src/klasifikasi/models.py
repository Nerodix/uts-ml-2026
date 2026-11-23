"""Perbandingan penanganan *class imbalance* dan pemodelan multi-keluarga (Tugas 2d).

Seluruh eksperimen dijalankan di dalam skema resmi 3-Fold Expanding-Window
Time CV Matang (Mei, Jun, Jul 2025) dengan fitur perilaku kausal Tugas 2c
(risiko historis lag-25 hari dihitung ulang ketat per fold memakai batas
``fold.train_end``):

1. **Perbandingan ≥3 Pendekatan Class Imbalance (pada LightGBM identik):**
   - Prior alami + ambang baku 0,5 (baseline)
   - *Class weighting* (``scale_pos_weight = n_neg / n_pos``)
   - *Random Undersampling* (rasio 1:10 di dalam fold latih saja)
   - *SMOTE Oversampling* (rasio 1:10 di dalam fold latih saja)
   - *Threshold tuning* F1-optimal & *cost-sensitive* Bayes (``p * nominal > 25.000``)
2. **Perbandingan 4 Keluarga Model + Ensemble Blend:**
   - Baseline ``LogisticRegression`` (OHE + standardisasi)
   - ``LightGBM`` (dituning di dalam Time CV)
   - ``CatBoost`` (kategori native, dituning di dalam Time CV)
   - ``XGBoost`` (``hist`` + kategori native, dituning di dalam Time CV)
   - ``Ensemble Log-Odds Blend`` (kombinasi konveks berbobot SLSQP)

Jalankan langsung:
    python kode/src/klasifikasi/models.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    FRAUD_REVIEW_COST_IDR,
    KLASIFIKASI_TEST_PATH,
    KLASIFIKASI_TRAIN_PATH,
    N_JOBS,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.klasifikasi.features import (  # noqa: E402
    SELECTED_TASK2_FEATURES,
    bangun_fitur_bebas_label,
)
from src.klasifikasi.preprocess import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    align_categorical_dtypes,
    muat_data_klasifikasi,
)
from src.klasifikasi.validation import FraudTimeFold, buat_fold_waktu_matang  # noqa: E402
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    configure_plotting,
    fraud_total_cost,
    label_batang,
    miring,
    save_json,
    set_global_seed,
    sumbu_koma,
    timer,
)

FEATURE_LIST = list(SELECTED_TASK2_FEATURES)
NUMERIC_FEATURES = [c for c in FEATURE_LIST if c not in CATEGORICAL_COLUMNS]


@dataclass
class PreparedFraudFold:
    """Satu fold validasi waktu matang yang fitur kausal bebas-labelnya diselaraskan."""

    fold: FraudTimeFold
    tr_df: pd.DataFrame
    va_df: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray
    nom_va: np.ndarray


def siapkan_fold_matang(train_feat: pd.DataFrame) -> list[PreparedFraudFold]:
    """Siapkan 3 fold matang dengan kategori diselaraskan ketat dari masing-masing fold latih."""
    folds = buat_fold_waktu_matang(train_feat)
    prepared: list[PreparedFraudFold] = []
    for f in folds:
        tr_raw = train_feat.iloc[f.train_idx]
        va_raw = train_feat.iloc[f.val_idx]
        tr_al, [va_al] = align_categorical_dtypes(tr_raw, [va_raw], cat_cols=CATEGORICAL_COLUMNS)
        prepared.append(
            PreparedFraudFold(
                fold=f,
                tr_df=tr_al,
                va_df=va_al,
                y_tr=tr_al[TARGET_COLUMN].to_numpy(dtype=np.int8),
                y_va=va_al[TARGET_COLUMN].to_numpy(dtype=np.int8),
                nom_va=va_al["nominal"].to_numpy(dtype=np.float64),
            )
        )
    return prepared


def hitung_metrik_prediksi(
    y_true: np.ndarray,
    prob: np.ndarray,
    nominal: np.ndarray,
    threshold_tetap: float | None = None,
) -> dict[str, float]:
    """Hitung PR-AUC, ROC-AUC, Brier, log-loss, F1, recall, dan total biaya."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y_true, dtype=np.int8)
    nom = np.asarray(nominal, dtype=np.float64)

    if threshold_tetap is not None:
        flg = (p >= threshold_tetap).astype(np.int8)
    else:
        flg = (p * nom > FRAUD_REVIEW_COST_IDR).astype(np.int8)

    cost = fraud_total_cost(y, flg, nom, review_cost=FRAUD_REVIEW_COST_IDR)
    cost_none = fraud_total_cost(y, np.zeros_like(y), nom, review_cost=FRAUD_REVIEW_COST_IDR)
    tp = int(((y == 1) & (flg == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "f1": float(f1_score(y, flg, zero_division=0)),
        "recall": float(tp / max(int(y.sum()), 1)),
        "flag_rate_pct": float(flg.mean() * 100.0),
        "total_cost_idr": float(cost),
        "cost_per_10k_idr": float(cost / len(y) * 10_000.0),
        "cost_reduction_pct": float((1.0 - cost / max(cost_none, 1.0)) * 100.0),
    }


def _ringkas_lintas_fold(per_fold: list[dict[str, float]]) -> dict[str, Any]:
    """Rangkum metrik rata-rata dan simpangan baku lintas 3 fold matang."""
    return {
        "mean_pr_auc": float(np.mean([r["pr_auc"] for r in per_fold])),
        "std_pr_auc": float(np.std([r["pr_auc"] for r in per_fold], ddof=1)),
        "mean_roc_auc": float(np.mean([r["roc_auc"] for r in per_fold])),
        "mean_brier": float(np.mean([r["brier"] for r in per_fold])),
        "mean_log_loss": float(np.mean([r["log_loss"] for r in per_fold])),
        "mean_f1": float(np.mean([r["f1"] for r in per_fold])),
        "mean_recall": float(np.mean([r["recall"] for r in per_fold])),
        "mean_flag_rate_pct": float(np.mean([r["flag_rate_pct"] for r in per_fold])),
        "total_cost_3fold_idr": float(np.sum([r["total_cost_idr"] for r in per_fold])),
        "mean_cost_per_10k_idr": float(np.mean([r["cost_per_10k_idr"] for r in per_fold])),
        "per_fold_pr_auc": [float(r["pr_auc"]) for r in per_fold],
        "per_fold_cost_idr": [float(r["total_cost_idr"]) for r in per_fold],
    }


def _base_lgbm_params(seed: int = SEED) -> dict[str, Any]:
    return {
        "objective": "binary",
        "n_estimators": 450,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }


def _encode_numeric_for_smote(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Konversi fitur kategorikal ke kode integer & imputasi median untuk SMOTE ketat dalam fold."""
    X_tr = tr_df[FEATURE_LIST].copy()
    X_va = va_df[FEATURE_LIST].copy()
    for c in CATEGORICAL_COLUMNS:
        X_tr[c] = X_tr[c].cat.codes.astype(np.float32)
        X_va[c] = X_va[c].cat.codes.astype(np.float32)
    med = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(med).fillna(0.0).to_numpy(dtype=np.float32)
    X_va = X_va.fillna(med).fillna(0.0).to_numpy(dtype=np.float32)
    return X_tr, X_va


def bandingkan_pendekatan_imbalance(
    folds: list[PreparedFraudFold],
    seed: int = SEED,
) -> dict[str, Any]:
    """Bandingkan 5 pendekatan *class imbalance* pada LightGBM identik di 3-Fold Time CV.

    1. ``natural_ambang_05``   : Tanpa penanganan imbalance, ambang baku 0,5.
    2. ``class_weighting``     : ``scale_pos_weight = n_neg / n_pos``, ambang 0,5.
    3. ``random_undersampling``: ``RandomUnderSampler`` (1:10) di dalam fold latih.
    4. ``smote_oversampling``  : ``SMOTE`` (1:10) di dalam fold latih.
    5. ``threshold_f1_tuning`` : Prior alami + ambang F1 optimal di dalam latih/OOF sebelumnya.
    6. ``cost_sensitive_bayes``: Prior alami + aturan biaya Bayes ``p * nominal > 25.000``.
    """
    m_nat05, m_cw, m_cw_bayes, m_rus, m_smote, m_f1, m_bayes = [], [], [], [], [], [], []

    # Ambang F1 diestimasi bertahap dari fold sebelumnya (untuk Fold 1 pakai estimasi in-fold konservatif 0,25)
    prev_probs: list[np.ndarray] = []
    prev_ys: list[np.ndarray] = []

    for pf in folds:
        X_tr_df = pf.tr_df[FEATURE_LIST]
        X_va_df = pf.va_df[FEATURE_LIST]

        # 1) Prior alami (juga dipakai untuk threshold tuning F1 & Bayes cost-sensitive)
        clf_nat = lgb.LGBMClassifier(**_base_lgbm_params(seed))
        clf_nat.fit(X_tr_df, pf.y_tr)
        p_nat = clf_nat.predict_proba(X_va_df)[:, 1]
        m_nat05.append(hitung_metrik_prediksi(pf.y_va, p_nat, pf.nom_va, threshold_tetap=0.5))
        m_bayes.append(hitung_metrik_prediksi(pf.y_va, p_nat, pf.nom_va, threshold_tetap=None))

        # Threshold tuning F1 tanpa mengintip fold validasi aktif
        if prev_probs:
            p_hist = np.concatenate(prev_probs)
            y_hist = np.concatenate(prev_ys)
            best_t, best_f1 = 0.25, -1.0
            for t in np.linspace(0.05, 0.70, 66):
                sc = f1_score(y_hist, (p_hist >= t).astype(np.int8), zero_division=0)
                if sc > best_f1:
                    best_f1, best_t = sc, float(t)
        else:
            best_t = 0.25
        m_f1.append(hitung_metrik_prediksi(pf.y_va, p_nat, pf.nom_va, threshold_tetap=best_t))
        prev_probs.append(p_nat)
        prev_ys.append(pf.y_va)

        # 2) Class Weighting (scale_pos_weight = n_neg / n_pos)
        spw = float((pf.y_tr == 0).sum() / max(int((pf.y_tr == 1).sum()), 1))
        cw_params = _base_lgbm_params(seed)
        cw_params["scale_pos_weight"] = spw
        clf_cw = lgb.LGBMClassifier(**cw_params)
        clf_cw.fit(X_tr_df, pf.y_tr)
        p_cw = clf_cw.predict_proba(X_va_df)[:, 1]
        m_cw.append(hitung_metrik_prediksi(pf.y_va, p_cw, pf.nom_va, threshold_tetap=0.5))
        m_cw_bayes.append(hitung_metrik_prediksi(pf.y_va, p_cw, pf.nom_va, threshold_tetap=None))

        # 3) Random Undersampling (1:10) di dalam fold latih
        rus = RandomUnderSampler(sampling_strategy=0.1, random_state=seed)
        idx_res, _ = rus.fit_resample(np.arange(len(pf.y_tr)).reshape(-1, 1), pf.y_tr)
        idx_res = idx_res.ravel()
        clf_rus = lgb.LGBMClassifier(**_base_lgbm_params(seed))
        clf_rus.fit(X_tr_df.iloc[idx_res], pf.y_tr[idx_res])
        p_rus = clf_rus.predict_proba(X_va_df)[:, 1]
        m_rus.append(hitung_metrik_prediksi(pf.y_va, p_rus, pf.nom_va, threshold_tetap=0.5))

        # 4) SMOTE Oversampling (1:10) pada matriks numerik ketat di dalam fold latih
        X_tr_num, X_va_num = _encode_numeric_for_smote(pf.tr_df, pf.va_df)
        smote = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed)
        X_sm, y_sm = smote.fit_resample(X_tr_num, pf.y_tr)
        clf_sm = lgb.LGBMClassifier(**_base_lgbm_params(seed))
        clf_sm.fit(X_sm, y_sm)
        p_sm = clf_sm.predict_proba(X_va_num)[:, 1]
        m_smote.append(hitung_metrik_prediksi(pf.y_va, p_sm, pf.nom_va, threshold_tetap=0.5))

    return {
        "natural_ambang_05": _ringkas_lintas_fold(m_nat05),
        "class_weighting_05": _ringkas_lintas_fold(m_cw),
        "class_weighting_bayes_raw": _ringkas_lintas_fold(m_cw_bayes),
        "random_undersampling_1_10": _ringkas_lintas_fold(m_rus),
        "smote_oversampling_1_10": _ringkas_lintas_fold(m_smote),
        "threshold_tuning_f1": _ringkas_lintas_fold(m_f1),
        "cost_sensitive_bayes": _ringkas_lintas_fold(m_bayes),
    }


def fit_predict_logreg(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    c_val: float = 0.2,
    seed: int = SEED,
) -> np.ndarray:
    """Latih baseline linear LogisticRegression dengan imputasi median, skala, dan OHE."""
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(C=c_val, max_iter=400, random_state=seed)),
        ]
    )
    pipe.fit(tr_df[FEATURE_LIST], tr_df[TARGET_COLUMN].to_numpy(dtype=np.int8))
    return pipe.predict_proba(va_df[FEATURE_LIST])[:, 1]


def fit_predict_lgbm(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    params: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    """Latih LightGBM biner dan kembalikan model serta probabilitas validasi."""
    clf = lgb.LGBMClassifier(**params)
    clf.fit(tr_df[FEATURE_LIST], tr_df[TARGET_COLUMN].to_numpy(dtype=np.int8), sample_weight=sample_weight)
    return clf, clf.predict_proba(va_df[FEATURE_LIST])[:, 1]


def fit_predict_catboost(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    params: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[CatBoostClassifier, np.ndarray]:
    """Latih CatBoost biner dengan kategori native yang aman terhadap nilai kosong."""
    X_tr = tr_df[FEATURE_LIST].copy()
    X_va = va_df[FEATURE_LIST].copy()
    for c in CATEGORICAL_COLUMNS:
        X_tr[c] = X_tr[c].astype(str).fillna("__NA__")
        X_va[c] = X_va[c].astype(str).fillna("__NA__")
    clf = CatBoostClassifier(**params)
    clf.fit(
        X_tr,
        tr_df[TARGET_COLUMN].to_numpy(dtype=np.int8),
        cat_features=list(CATEGORICAL_COLUMNS),
        sample_weight=sample_weight,
        verbose=False,
    )
    return clf, clf.predict_proba(X_va)[:, 1]


def fit_predict_xgboost(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    params: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[xgb.XGBClassifier, np.ndarray]:
    """Latih XGBoost histogram biner dengan dukungan kategori native."""
    clf = xgb.XGBClassifier(**params)
    clf.fit(tr_df[FEATURE_LIST], tr_df[TARGET_COLUMN].to_numpy(dtype=np.int8), sample_weight=sample_weight)
    return clf, clf.predict_proba(va_df[FEATURE_LIST])[:, 1]


def _optimasi_bobot_blend_logit(
    preds_list: list[np.ndarray],
    y_true: np.ndarray,
) -> np.ndarray:
    """Cari bobot konveks optimal di ruang log-odds (meminimalkan log-loss pada OOF)."""
    logits = np.column_stack([logit(np.clip(p, 1e-6, 1.0 - 1e-6)) for p in preds_list])
    k = logits.shape[1]

    def _obj(w: np.ndarray) -> float:
        p_blend = np.clip(expit(logits @ w), 1e-7, 1.0 - 1e-7)
        # Skalakan 1e3 agar toleransi konvergensi SLSQP tidak terhenti pada laju positif 0,65%
        return float(log_loss(y_true, p_blend) * 1e3)

    w0 = np.ones(k) / k
    res = minimize(
        _obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"ftol": 1e-12, "maxiter": 200},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


def gabung_prediksi_logit(preds_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Gabungkan daftar probabilitas memakai kombinasi konveks di ruang log-odds."""
    logits = np.column_stack([logit(np.clip(p, 1e-6, 1.0 - 1e-6)) for p in preds_list])
    return expit(logits @ np.asarray(weights, dtype=np.float64))


def bandingkan_dan_tuning_model(
    folds: list[PreparedFraudFold],
    seed: int = SEED,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]], dict[str, Any]]:
    """Tuning dan bandingkan 4 keluarga model + Ensemble di dalam 3-Fold Mature Time CV."""
    oof_preds: dict[str, list[np.ndarray]] = {}
    ringkasan_model: dict[str, Any] = {}
    best_configs: dict[str, Any] = {}

    # 1) Baseline Logistic Regression (tuning C)
    with timer("2d.2 Baseline Logistic Regression (tuning C)"):
        best_lr_pr, best_c, best_lr_preds, best_lr_mets = -1.0, 0.2, [], []
        for c_val in (0.05, 0.2, 1.0):
            ps, ms = [], []
            for pf in folds:
                p = fit_predict_logreg(pf.tr_df, pf.va_df, c_val=c_val, seed=seed)
                ps.append(p)
                ms.append(hitung_metrik_prediksi(pf.y_va, p, pf.nom_va))
            sc = float(np.mean([r["pr_auc"] for r in ms]))
            if sc > best_lr_pr:
                best_lr_pr, best_c, best_lr_preds, best_lr_mets = sc, c_val, ps, ms
        oof_preds["logreg"] = best_lr_preds
        ringkasan_model["logreg"] = _ringkas_lintas_fold(best_lr_mets)
        best_configs["logreg"] = {"C": best_c}
        print(f"  LogReg (C={best_c}): PR-AUC={ringkasan_model['logreg']['mean_pr_auc']:.4f}, Biaya/10k={ringkasan_model['logreg']['mean_cost_per_10k_idr']:,.0f}")

    # 2) LightGBM (tuning dalam Time CV)
    with timer("2d.3 LightGBM (tuning dalam 3-Fold Time CV)"):
        lgb_grid = [
            {"num_leaves": 31, "min_child_samples": 40, "colsample_bytree": 0.75, "reg_lambda": 5.0, "n_estimators": 500, "learning_rate": 0.03},
            {"num_leaves": 63, "min_child_samples": 60, "colsample_bytree": 0.65, "reg_lambda": 10.0, "n_estimators": 550, "learning_rate": 0.03},
            {"num_leaves": 23, "min_child_samples": 30, "colsample_bytree": 0.80, "reg_lambda": 8.0, "n_estimators": 600, "learning_rate": 0.025},
        ]
        best_pr, best_cfg, best_ps, best_ms = -1.0, {}, [], []
        for cfg in lgb_grid:
            full_cfg = {
                "objective": "binary",
                "subsample": 0.8,
                "subsample_freq": 1,
                "reg_alpha": 1.0,
                "random_state": seed,
                "n_jobs": N_JOBS,
                "verbose": -1,
                **cfg,
            }
            ps, ms = [], []
            for pf in folds:
                _, p = fit_predict_lgbm(pf.tr_df, pf.va_df, full_cfg)
                ps.append(p)
                ms.append(hitung_metrik_prediksi(pf.y_va, p, pf.nom_va))
            sc = float(np.mean([r["pr_auc"] for r in ms]))
            if sc > best_pr:
                best_pr, best_cfg, best_ps, best_ms = sc, full_cfg, ps, ms
        oof_preds["lgbm"] = best_ps
        ringkasan_model["lgbm"] = _ringkas_lintas_fold(best_ms)
        best_configs["lgbm"] = {k: v for k, v in best_cfg.items() if k != "n_jobs"}
        print(f"  LightGBM terbaik: PR-AUC={ringkasan_model['lgbm']['mean_pr_auc']:.4f}, Biaya/10k={ringkasan_model['lgbm']['mean_cost_per_10k_idr']:,.0f}")

    # 3) CatBoost (tuning dalam Time CV)
    with timer("2d.4 CatBoost (tuning dalam 3-Fold Time CV)"):
        cb_grid = [
            {"depth": 6, "l2_leaf_reg": 6.0, "iterations": 750, "learning_rate": 0.04},
            {"depth": 5, "l2_leaf_reg": 10.0, "iterations": 850, "learning_rate": 0.035},
        ]
        best_pr, best_cfg, best_ps, best_ms = -1.0, {}, [], []
        for cfg in cb_grid:
            full_cfg = {
                "loss_function": "Logloss",
                "random_seed": seed,
                "thread_count": N_JOBS,
                "allow_writing_files": False,
                **cfg,
            }
            ps, ms = [], []
            for pf in folds:
                _, p = fit_predict_catboost(pf.tr_df, pf.va_df, full_cfg)
                ps.append(p)
                ms.append(hitung_metrik_prediksi(pf.y_va, p, pf.nom_va))
            sc = float(np.mean([r["pr_auc"] for r in ms]))
            if sc > best_pr:
                best_pr, best_cfg, best_ps, best_ms = sc, full_cfg, ps, ms
        oof_preds["catboost"] = best_ps
        ringkasan_model["catboost"] = _ringkas_lintas_fold(best_ms)
        best_configs["catboost"] = {k: v for k, v in best_cfg.items() if k != "thread_count"}
        print(f"  CatBoost terbaik: PR-AUC={ringkasan_model['catboost']['mean_pr_auc']:.4f}, Biaya/10k={ringkasan_model['catboost']['mean_cost_per_10k_idr']:,.0f}")

    # 4) XGBoost (tuning dalam Time CV)
    with timer("2d.5 XGBoost (tuning dalam 3-Fold Time CV)"):
        xgb_grid = [
            {"max_depth": 5, "min_child_weight": 20, "colsample_bytree": 0.75, "reg_lambda": 5.0, "n_estimators": 500, "learning_rate": 0.03},
            {"max_depth": 6, "min_child_weight": 35, "colsample_bytree": 0.70, "reg_lambda": 10.0, "n_estimators": 550, "learning_rate": 0.03},
        ]
        best_pr, best_cfg, best_ps, best_ms = -1.0, {}, [], []
        for cfg in xgb_grid:
            full_cfg = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "enable_categorical": True,
                "subsample": 0.8,
                "reg_alpha": 1.0,
                "random_state": seed,
                "n_jobs": N_JOBS,
                **cfg,
            }
            ps, ms = [], []
            for pf in folds:
                _, p = fit_predict_xgboost(pf.tr_df, pf.va_df, full_cfg)
                ps.append(p)
                ms.append(hitung_metrik_prediksi(pf.y_va, p, pf.nom_va))
            sc = float(np.mean([r["pr_auc"] for r in ms]))
            if sc > best_pr:
                best_pr, best_cfg, best_ps, best_ms = sc, full_cfg, ps, ms
        oof_preds["xgboost"] = best_ps
        ringkasan_model["xgboost"] = _ringkas_lintas_fold(best_ms)
        best_configs["xgboost"] = {k: v for k, v in best_cfg.items() if k != "n_jobs"}
        print(f"  XGBoost terbaik : PR-AUC={ringkasan_model['xgboost']['mean_pr_auc']:.4f}, Biaya/10k={ringkasan_model['xgboost']['mean_cost_per_10k_idr']:,.0f}")

    # 5) Ensemble Log-Odds Blend (bobot OOS Fold 1-2 -> Fold 3; bobot final pada seluruh 3 fold OOF)
    blend_members = ("lgbm", "catboost", "xgboost")
    y_f12 = np.concatenate([folds[0].y_va, folds[1].y_va])
    p_f12 = [np.concatenate([oof_preds[m][0], oof_preds[m][1]]) for m in blend_members]
    w_oos = _optimasi_bobot_blend_logit(p_f12, y_f12)

    y_all = np.concatenate([pf.y_va for pf in folds])
    p_all = [np.concatenate(oof_preds[m]) for m in blend_members]
    w_final = _optimasi_bobot_blend_logit(p_all, y_all)

    # Evaluasi jujur: Fold 1-2 pakai bobot seragam (pra-optimasi), Fold 3 pakai bobot OOS dari Fold 1-2
    ens_preds, ens_mets = [], []
    for idx, pf in enumerate(folds):
        w_use = w_oos if idx == 2 else np.ones(len(blend_members)) / len(blend_members)
        p_e = gabung_prediksi_logit([oof_preds[m][idx] for m in blend_members], w_use)
        ens_preds.append(p_e)
        ens_mets.append(hitung_metrik_prediksi(pf.y_va, p_e, pf.nom_va))

    oof_preds["ensemble"] = ens_preds
    ringkasan_model["ensemble"] = _ringkas_lintas_fold(ens_mets)
    best_configs["ensemble"] = {
        "members": list(blend_members),
        "weights_oos_f12": {m: float(w) for m, w in zip(blend_members, w_oos)},
        "weights_final_all3": {m: float(w) for m, w in zip(blend_members, w_final)},
    }
    print(
        f"  Ensemble Blend  : PR-AUC={ringkasan_model['ensemble']['mean_pr_auc']:.4f} "
        f"(F3 OOS={ens_mets[2]['pr_auc']:.4f}), Biaya/10k={ringkasan_model['ensemble']['mean_cost_per_10k_idr']:,.0f} | "
        f"bobot={best_configs['ensemble']['weights_final_all3']}"
    )
    return ringkasan_model, oof_preds, best_configs


def simpan_plot_imbalance_dan_model(
    imbalance: dict[str, Any],
    models: dict[str, Any],
    output_path: Path = FIGURES_DIR / "task2_2d_imbalance_and_models.png",
) -> Path:
    """Gambar figur 2 panel perbandingan pendekatan imbalance & keluarga model Tugas 2d."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.85))

    # Panel A: Perbandingan Pendekatan Class Imbalance (PR-AUC & Biaya/10k transaksi)
    ax0 = axes[0]
    imb_items = [
        ("natural_ambang_05", "Alami\n(t=0,5)"),
        ("class_weighting_05", "Class\nweight"),
        ("random_undersampling_1_10", "Under-\nsample"),
        ("smote_oversampling_1_10", "SMOTE\n(1:10)"),
        ("threshold_tuning_f1", "Tuning\nF1"),
        ("cost_sensitive_bayes", "Bayes\nbiaya"),
    ]
    x0 = np.arange(len(imb_items))
    w = 0.38
    pr_i = [imbalance[k]["mean_pr_auc"] for k, _ in imb_items]
    cost_i = [imbalance[k]["mean_cost_per_10k_idr"] / 1e6 for k, _ in imb_items]

    b1 = ax0.bar(x0 - w / 2, pr_i, width=w, color="#1f77b4", label="PR-AUC")
    ax0.set_xticks(x0)
    ax0.set_xticklabels([lbl for _, lbl in imb_items], fontsize=5.8)
    ax0.set_ylim(0, max(pr_i) * 1.25)
    ax0.set_ylabel("PR-AUC")
    sumbu_koma(ax0, "y")
    label_batang(ax0, b1, desimal=3, ukuran=5.2, putar=True)

    ax0_r = ax0.twinx()
    ax0_r.spines["top"].set_visible(False)
    ax0_r.grid(False)
    b2 = ax0_r.bar(x0 + w / 2, cost_i, width=w, color="#e6550d", alpha=0.82, label="Biaya/10k (jt Rp)")
    ax0_r.set_ylim(0, max(cost_i) * 1.32)
    ax0_r.set_ylabel("Biaya / 10 rb trx (juta Rp)", color="#b33c00")
    ax0_r.tick_params(axis="y", colors="#b33c00")
    sumbu_koma(ax0_r, "y")
    for bar, val in zip(b2, cost_i):
        ax0_r.annotate(
            angka_id(val, 1),
            (bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 1.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.2,
            color="#b33c00",
        )
    ax0.set_title(f"(a) Penanganan {miring('class imbalance')}")
    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = ax0_r.get_legend_handles_labels()
    ax0.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=5.6)

    # Panel B: Perbandingan 4 Keluarga Model + Ensemble (PR-AUC & Biaya/10k)
    ax1 = axes[1]
    mod_items = [
        ("logreg", "LogReg\n(dasar)"),
        ("xgboost", "XGBoost"),
        ("catboost", "CatBoost"),
        ("lgbm", "LightGBM"),
        ("ensemble", "Ensemble\nBlend"),
    ]
    x1 = np.arange(len(mod_items))
    pr_m = [models[k]["mean_pr_auc"] for k, _ in mod_items]
    cost_m = [models[k]["mean_cost_per_10k_idr"] / 1e6 for k, _ in mod_items]
    cols_m = ["#9e9e9e", "#6baed6", "#41ab5d", "#2171b5", "#08519c"]
    b_m = ax1.bar(x1, pr_m, color=cols_m, width=0.56)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl in mod_items], fontsize=6.2)
    ax1.set_ylim(0, max(pr_m) * 1.22)
    ax1.set_ylabel("PR-AUC (3-Fold Time CV)")
    ax1.set_title("(b) Perbandingan model & biaya Bayes")
    sumbu_koma(ax1, "y")
    for bar, pr_v, c_v in zip(b_m, pr_m, cost_m):
        ax1.annotate(
            f"{angka_id(pr_v, 3)}\n({angka_id(c_v, 1)} jt)",
            (bar.get_x() + bar.get_width() / 2, pr_v),
            xytext=(0, 2.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.4,
        )

    fig.tight_layout(w_pad=1.6)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def jalankan_pemodelan_klasifikasi(save_json_flag: bool = True) -> dict[str, Any]:
    """Jalankan eksperimen imbalance 2d, tuning multi-model, dan simpan cache OOF."""
    set_global_seed(SEED)
    ensure_dirs()

    with timer("Muat data & bangun fitur kausal 2c"):
        train_raw, _ = muat_data_klasifikasi(KLASIFIKASI_TRAIN_PATH, KLASIFIKASI_TEST_PATH)
        train_feat, _ = bangun_fitur_bebas_label(train_raw, None)
        folds = siapkan_fold_matang(train_feat)

    with timer("2d.1 Perbandingan pendekatan class imbalance"):
        hasil_imb = bandingkan_pendekatan_imbalance(folds, seed=SEED)
        for nama, met in hasil_imb.items():
            print(
                f"  {nama:26s}: PR-AUC={met['mean_pr_auc']:.4f}, F1={met['mean_f1']:.4f}, "
                f"Brier={met['mean_brier']:.5f}, Biaya/10k={met['mean_cost_per_10k_idr']:,.0f}"
            )

    hasil_mod, oof_preds, best_configs = bandingkan_dan_tuning_model(folds, seed=SEED)
    simpan_plot_imbalance_dan_model(hasil_imb, hasil_mod)

    # Simpan OOF cache terstruktur untuk tahap kalibrasi 2f & analisis 2g
    cache_payload: dict[str, Any] = {
        "y_f1": folds[0].y_va,
        "y_f2": folds[1].y_va,
        "y_f3": folds[2].y_va,
        "nom_f1": folds[0].nom_va,
        "nom_f2": folds[1].nom_va,
        "nom_f3": folds[2].nom_va,
        "idx_f1": folds[0].fold.val_idx,
        "idx_f2": folds[1].fold.val_idx,
        "idx_f3": folds[2].fold.val_idx,
    }
    for m_name, plist in oof_preds.items():
        for i, arr in enumerate(plist, start=1):
            cache_payload[f"{m_name}_f{i}"] = arr.astype(np.float64)
    np.savez_compressed(CACHE_DIR / "task2_oof.npz", **cache_payload)

    hasil = {
        "n_features": len(FEATURE_LIST),
        "imbalance_comparison": hasil_imb,
        "model_comparison": hasil_mod,
        "best_configs": best_configs,
    }
    if save_json_flag:
        save_json(hasil, RESULTS_DIR / "task2_models.json")
        print("[SAVED] task2_models.json & cache/task2_oof.npz")
    return hasil


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2d class imbalance comparison and model tuning.")
    parser.add_argument("--no-save-json", action="store_true", help="Skip saving task2_models.json.")
    args = parser.parse_args()
    jalankan_pemodelan_klasifikasi(save_json_flag=not args.no_save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
