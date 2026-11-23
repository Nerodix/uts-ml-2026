"""Kalibrasi probabilitas dan aturan keputusan berbasis biaya Bayes (Tugas 2f).

Membandingkan empat kondisi kalibrasi secara kronologis murni (fit kalibrator
pada OOF Fold 1–2 Mei–Jun, evaluasi di luar sampel pada Fold 3 Jul):
1. ``uncalibrated``     : Probabilitas mentah Ensemble Log-Odds Blend.
2. ``platt_scaling``    : Kalibrasi sigmoid pada skor log-odds (monoton ketat, PR-AUC utuh).
3. ``isotonic``         : Regresi isotonik non-parametrik (``out_of_bounds='clip'``).
4. ``beta_calibration`` : Kalibrasi Beta 3-parameter ``a*log(p) + b*log(1-p) + c``
   (Kull dkk., 2017; monoton ketat dan lentur untuk probabilitas sangat timpang).

Setelah memilih kalibrator terbaik, mengevaluasi aturan keputusan berbasis biaya
Bayes optimal ``tandai = 1 iff p * nominal > Rp 25.000`` dibandingkan dengan:
- Tandai semua (``flag_all``)
- Tandai nol (``flag_none``)
- Ambang tetap 0,5 & ambang tetap F1-optimal (dari Fold 1–2)
- Aturan Bayes sebelum kalibrasi & batas bawah teoritis *Oracle*.

Jalankan langsung:
    python kode/src/klasifikasi/calibration.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    FRAUD_REVIEW_COST_IDR,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    configure_plotting,
    fraud_total_cost,
    miring,
    save_json,
    set_global_seed,
    sumbu_koma,
    timer,
)


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


class PlattCalibrator:
    """Kalibrasi sigmoid (Platt scaling) di ruang log-odds ``a * logit(p) + b``."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e4, max_iter=500, random_state=SEED)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        z = logit(_clip_prob(probs)).reshape(-1, 1)
        self._lr.fit(z, np.asarray(y, dtype=np.int8))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        z = logit(_clip_prob(probs)).reshape(-1, 1)
        return _clip_prob(self._lr.predict_proba(z)[:, 1])

    def params(self) -> dict[str, float]:
        return {
            "slope": float(self._lr.coef_[0, 0]),
            "intercept": float(self._lr.intercept_[0]),
        }


class BetaCalibrator:
    """Kalibrasi Beta 3-parameter ``logit(p_cal) = a * log(p) + b * (-log(1-p)) + c``."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e4, max_iter=500, random_state=SEED)

    @staticmethod
    def _features(probs: np.ndarray) -> np.ndarray:
        p = _clip_prob(probs)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, probs: np.ndarray, y: np.ndarray) -> BetaCalibrator:
        self._lr.fit(self._features(probs), np.asarray(y, dtype=np.int8))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return _clip_prob(self._lr.predict_proba(self._features(probs))[:, 1])

    def params(self) -> dict[str, float]:
        return {
            "a_log_p": float(self._lr.coef_[0, 0]),
            "b_neg_log_1mp": float(self._lr.coef_[0, 1]),
            "intercept": float(self._lr.intercept_[0]),
        }


class IsotonicCalibrator:
    """Kalibrasi non-parametrik monoton menggunakan ``IsotonicRegression``."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")

    def fit(self, probs: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self._iso.fit(_clip_prob(probs), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return _clip_prob(self._iso.predict(_clip_prob(probs)))

    def params(self) -> dict[str, int]:
        return {"n_thresholds": int(len(self._iso.X_thresholds_))}


def hitung_ece_kuantil(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    """Hitung *Expected Calibration Error* (ECE) berbasis kuantil (adaptif untuk kelas timpang)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = _clip_prob(prob)
    qs = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(qs) <= 2:
        return float(abs(np.mean(p) - np.mean(y)))
    bin_ids = np.digitize(p, qs[1:-1], right=True)
    ece = 0.0
    n = len(y)
    for b in range(len(qs) - 1):
        m = bin_ids == b
        if m.any():
            ece += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ece)


def hitung_kurva_reliabilitas_tail(
    y_true: np.ndarray,
    prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bangun titik diagram reliabilitas yang resolusinya rapat di ekor risiko tinggi."""
    y = np.asarray(y_true, dtype=np.float64)
    p = _clip_prob(prob)
    # Tepi probabilitas logaritmik agar kelas 0,65% maupun ekor 0,1–1,0 terlihat jelas
    edges = np.array([0.0, 0.002, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.01])
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 15:
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
    return np.asarray(xs), np.asarray(ys)


def buat_kalibrator(nama: str) -> PlattCalibrator | BetaCalibrator | IsotonicCalibrator | None:
    """Bangun instans kalibrator berdasarkan nama metode."""
    if nama == "platt_scaling":
        return PlattCalibrator()
    if nama == "beta_calibration":
        return BetaCalibrator()
    if nama == "isotonic":
        return IsotonicCalibrator()
    return None


def aturan_keputusan_bayes(
    prob: np.ndarray,
    nominal: np.ndarray,
    review_cost: float = FRAUD_REVIEW_COST_IDR,
    pengali_ambang: float = 1.0,
) -> np.ndarray:
    """Aturan keputusan berbasis biaya Bayes: tandai iff ``p * nominal > k * review_cost``.

    Secara teoritis untuk probabilitas terkalibrasi sempurna, ``pengali_ambang = 1,0``
    meminimalkan ekspektasi biaya total karena kerugian lolos = ``p * nominal`` dan
    biaya tinjauan = ``review_cost`` (Rp 25.000).
    """
    p = _clip_prob(prob)
    nom = np.asarray(nominal, dtype=np.float64)
    return (p * nom > pengali_ambang * review_cost).astype(np.int8)


def _ringkas_kalibrasi(
    y: np.ndarray,
    prob: np.ndarray,
    nom: np.ndarray,
    pengali: float = 1.0,
) -> dict[str, float]:
    p = _clip_prob(prob)
    flg = aturan_keputusan_bayes(p, nom, pengali_ambang=pengali)
    cost = fraud_total_cost(y, flg, nom, review_cost=FRAUD_REVIEW_COST_IDR)
    cost_none = fraud_total_cost(y, np.zeros_like(y), nom, review_cost=FRAUD_REVIEW_COST_IDR)
    tp = int(((y == 1) & (flg == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece_15bin": hitung_ece_kuantil(y, p, n_bins=15),
        "mean_prob_pct": float(p.mean() * 100.0),
        "obs_rate_pct": float(y.mean() * 100.0),
        "ratio_pred_to_obs": float(p.mean() / max(float(y.mean()), 1e-12)),
        "total_cost_idr": float(cost),
        "cost_reduction_vs_none_pct": float((1.0 - cost / max(cost_none, 1.0)) * 100.0),
        "n_flagged": int(flg.sum()),
        "flag_rate_pct": float(flg.mean() * 100.0),
        "recall": float(tp / max(int(y.sum()), 1)),
    }


def _ringkas_aturan_keputusan(
    y: np.ndarray,
    flg: np.ndarray,
    nom: np.ndarray,
) -> dict[str, float]:
    cost = fraud_total_cost(y, flg, nom, review_cost=FRAUD_REVIEW_COST_IDR)
    missed = float(nom[(y == 1) & (flg == 0)].sum())
    review = float(FRAUD_REVIEW_COST_IDR * (flg == 1).sum())
    tp = int(((y == 1) & (flg == 1)).sum())
    fp = int(((y == 0) & (flg == 1)).sum())
    return {
        "total_cost_idr": float(cost),
        "missed_fraud_loss_idr": missed,
        "review_cost_idr": review,
        "n_flagged": int(flg.sum()),
        "flag_rate_pct": float(flg.mean() * 100.0),
        "recall": float(tp / max(int(y.sum()), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "f1": float(f1_score(y, flg, zero_division=0)),
    }


def simpan_plot_kalibrasi_dan_biaya(
    y_f3: np.ndarray,
    p_raw: np.ndarray,
    p_best: np.ndarray,
    best_name: str,
    decision_comp: dict[str, dict[str, float]],
    output_path: Path = FIGURES_DIR / "task2_2f_calibration_and_cost.png",
) -> Path:
    """Gambar figur 2 panel diagram reliabilitas & perbandingan total biaya keputusan 2f."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.75))

    # Panel A: Reliability Diagram (sebelum vs sesudah kalibrasi pada Fold 3 Jul)
    ax0 = axes[0]
    x_raw, y_raw = hitung_kurva_reliabilitas_tail(y_f3, p_raw)
    x_cal, y_cal = hitung_kurva_reliabilitas_tail(y_f3, p_best)
    ax0.plot([0, 1], [0, 1], "--", color="#555555", linewidth=0.9, label="Kalibrasi sempurna")
    ax0.plot(x_raw, y_raw, "o-", color="#d62728", linewidth=1.2, label="Sebelum kalibrasi")
    lbl_cal = best_name.replace("_", " ").title()
    ax0.plot(x_cal, y_cal, "s-", color="#2ca02c", linewidth=1.3, label=f"Sesudah ({lbl_cal})")
    ax0.set_xlim(0, 1.0)
    ax0.set_ylim(0, 1.0)
    ax0.set_xlabel("Rata-rata probabilitas prediksi")
    ax0.set_ylabel("Proporsi fraud aktual (Fold 3 Jul)")
    ax0.set_title("(a) Diagram reliabilitas probabilitas")
    sumbu_koma(ax0, "xy")
    ax0.legend(loc="upper left", fontsize=5.8)

    # Panel B: Perbandingan Total Biaya Fold 3 (juta Rp) lintas aturan keputusan
    ax1 = axes[1]
    items = [
        ("tandai_semua", "Tandai\nsemua", "#8c8c8c"),
        ("tandai_nol", "Tandai\nnol", "#d62728"),
        ("ambang_tetap_05", "Ambang\np≥0,5", "#ff7f0e"),
        ("ambang_tetap_f1", "Ambang\nF1", "#fdae6b"),
        ("aturan_bayes_uncal", "Bayes\nmentah", "#6baed6"),
        ("aturan_bayes_terkalibrasi", "Bayes\nkalibrasi", "#2ca02c"),
        ("oracle_teoritis", "Oracle", "#1b7837"),
    ]
    x1 = np.arange(len(items))
    vals = [decision_comp[k]["total_cost_idr"] / 1e6 for k, _, _ in items]
    cols = [c for _, _, c in items]
    bars = ax1.bar(x1, vals, color=cols, width=0.60)
    ax1.set_yscale("log")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl, _ in items], fontsize=5.5)
    ax1.set_ylim(max(min(vals) * 0.45, 1.0), max(vals) * 2.8)
    ax1.set_ylabel("Total biaya Fold 3 (juta Rp, skala log)")
    ax1.set_title(f"(b) Total biaya aturan {miring('Bayes')} vs pembanding")
    for bar, val in zip(bars, vals):
        ax1.annotate(
            angka_id(val, 1),
            (bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 2.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.4,
        )

    fig.tight_layout(w_pad=1.5)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def jalankan_kalibrasi_dan_keputusan(save_json_flag: bool = True) -> dict[str, Any]:
    """Jalankan evaluasi kalibrasi kronologis (Fit F1–F2 -> Eval F3) & aturan biaya 2f."""
    set_global_seed(SEED)
    ensure_dirs()

    oof_path = CACHE_DIR / "task2_oof.npz"
    if not oof_path.exists():
        raise FileNotFoundError(
            "Cache OOF tidak ditemukan. Jalankan `python kode/src/klasifikasi/models.py` terlebih dahulu."
        )

    with timer("2f.1 Kalibrasi kronologis (Fit Mei–Jun -> Eval Jul)"):
        oof = np.load(oof_path)
        p_cal_fit = np.concatenate([oof["ensemble_f1"], oof["ensemble_f2"]])
        y_cal_fit = np.concatenate([oof["y_f1"], oof["y_f2"]])
        nom_cal_fit = np.concatenate([oof["nom_f1"], oof["nom_f2"]])

        p_f3 = oof["ensemble_f3"]
        y_f3 = oof["y_f3"]
        nom_f3 = oof["nom_f3"]

        hasil_cal: dict[str, dict[str, Any]] = {}
        preds_f3: dict[str, np.ndarray] = {"uncalibrated": _clip_prob(p_f3)}
        hasil_cal["uncalibrated"] = _ringkas_kalibrasi(y_f3, p_f3, nom_f3)

        for nama in ("platt_scaling", "beta_calibration", "isotonic"):
            cal = buat_kalibrator(nama)
            assert cal is not None
            cal.fit(p_cal_fit, y_cal_fit)
            p_eval = cal.predict(p_f3)
            preds_f3[nama] = p_eval
            met = _ringkas_kalibrasi(y_f3, p_eval, nom_f3)
            met["params_f12"] = cal.params()
            hasil_cal[nama] = met

        for nama, met in hasil_cal.items():
            print(
                f"  {nama:18s}: PR-AUC={met['pr_auc']:.4f} | Brier={met['brier']:.5f} | "
                f"LogLoss={met['log_loss']:.5f} | ECE={met['ece_15bin']:.5f} | "
                f"Biaya F3={met['total_cost_idr']/1e6:.2f}M Rp"
            )

        # Pilih kalibrator terbaik memakai strictly proper scoring rule (log-loss & Brier)
        # di antara metode yang menjaga PR-AUC (tidak turun >0,002 akibat step-function ties)
        pr_ref = hasil_cal["uncalibrated"]["pr_auc"]
        kandidat_baik = [
            n for n in ("uncalibrated", "platt_scaling", "beta_calibration", "isotonic")
            if hasil_cal[n]["pr_auc"] >= pr_ref - 0.002
        ]
        terbaik_nama = min(
            kandidat_baik,
            key=lambda n: (hasil_cal[n]["log_loss"], hasil_cal[n]["brier"]),
        )
        p_f3_best = preds_f3[terbaik_nama]
        print(f"  Kalibrator terpilih: {terbaik_nama}")

    with timer("2f.2 Evaluasi aturan keputusan berbasis biaya vs baseline"):
        # Cari ambang F1 optimal pada Fold 1–2 (bebas mengintip Fold 3)
        best_t_f1, best_f1 = 0.25, -1.0
        for t in np.linspace(0.05, 0.70, 66):
            sc = f1_score(y_cal_fit, (p_cal_fit >= t).astype(np.int8), zero_division=0)
            if sc > best_f1:
                best_f1, best_t_f1 = sc, float(t)

        # Verifikasi pengali empiris k di sekitar 1,0 pada Fold 1–2 untuk membandingkan dengan teori k=1,0
        cal_tmp = buat_kalibrator(terbaik_nama)
        p_f12_cal = cal_tmp.fit(p_cal_fit, y_cal_fit).predict(p_cal_fit) if cal_tmp is not None else _clip_prob(p_cal_fit)
        best_k, best_k_cost = 1.0, float("inf")
        for k_cand in np.linspace(0.75, 1.25, 21):
            flg_c = aturan_keputusan_bayes(p_f12_cal, nom_cal_fit, pengali_ambang=float(k_cand))
            c_val = fraud_total_cost(y_cal_fit, flg_c, nom_cal_fit)
            if c_val < best_k_cost:
                best_k_cost, best_k = c_val, float(k_cand)

        flg_all = np.ones_like(y_f3, dtype=np.int8)
        flg_none = np.zeros_like(y_f3, dtype=np.int8)
        flg_t05 = (p_f3_best >= 0.5).astype(np.int8)
        flg_tf1 = (p_f3_best >= best_t_f1).astype(np.int8)
        flg_bayes_raw = aturan_keputusan_bayes(p_f3, nom_f3, pengali_ambang=1.0)
        flg_bayes_cal = aturan_keputusan_bayes(p_f3_best, nom_f3, pengali_ambang=1.0)
        flg_bayes_k = aturan_keputusan_bayes(p_f3_best, nom_f3, pengali_ambang=best_k)
        flg_oracle = ((y_f3 == 1) & (nom_f3 > FRAUD_REVIEW_COST_IDR)).astype(np.int8)

        decision_comp = {
            "tandai_semua": _ringkas_aturan_keputusan(y_f3, flg_all, nom_f3),
            "tandai_nol": _ringkas_aturan_keputusan(y_f3, flg_none, nom_f3),
            "ambang_tetap_05": _ringkas_aturan_keputusan(y_f3, flg_t05, nom_f3),
            "ambang_tetap_f1": _ringkas_aturan_keputusan(y_f3, flg_tf1, nom_f3),
            "aturan_bayes_uncal": _ringkas_aturan_keputusan(y_f3, flg_bayes_raw, nom_f3),
            "aturan_bayes_terkalibrasi": _ringkas_aturan_keputusan(y_f3, flg_bayes_cal, nom_f3),
            "aturan_bayes_tuned_k": _ringkas_aturan_keputusan(y_f3, flg_bayes_k, nom_f3),
            "oracle_teoritis": _ringkas_aturan_keputusan(y_f3, flg_oracle, nom_f3),
        }
        for nama, met in decision_comp.items():
            print(
                f"  {nama:26s}: Biaya={met['total_cost_idr']/1e6:7.2f}M Rp "
                f"(lolos={met['missed_fraud_loss_idr']/1e6:6.2f}M, review={met['review_cost_idr']/1e6:6.2f}M, "
                f"flag={met['flag_rate_pct']:5.2f}%, rec={met['recall']*100:5.1f}%)"
            )

        # Fit kalibrator final pada seluruh 3 fold OOF matang (Mei–Jul) untuk produksi Sep–Okt
        p_all3 = np.concatenate([oof["ensemble_f1"], oof["ensemble_f2"], oof["ensemble_f3"]])
        y_all3 = np.concatenate([oof["y_f1"], oof["y_f2"], oof["y_f3"]])
        cal_final = buat_kalibrator(terbaik_nama)
        final_params = cal_final.fit(p_all3, y_all3).params() if cal_final is not None else {}

        # Simpan prediksi Fold 3 terkalibrasi untuk pemecahan error Tugas 2g
        np.savez_compressed(
            CACHE_DIR / "task2_fold3_calibrated.npz",
            idx_f3=oof["idx_f3"],
            y_f3=y_f3,
            nom_f3=nom_f3,
            prob_raw=p_f3,
            prob_cal=p_f3_best,
            tandai=flg_bayes_cal,
        )

    simpan_plot_kalibrasi_dan_biaya(y_f3, p_f3, p_f3_best, terbaik_nama, decision_comp)

    c_bayes = decision_comp["aturan_bayes_terkalibrasi"]["total_cost_idr"]
    c_none = decision_comp["tandai_nol"]["total_cost_idr"]
    c_all = decision_comp["tandai_semua"]["total_cost_idr"]
    hasil = {
        "split_kalibrasi": {
            "fit_period": "Fold 1–2 (Mei–Jun 2025)",
            "eval_period": "Fold 3 (Jul 2025)",
            "n_fit": int(len(y_cal_fit)),
            "n_eval": int(len(y_f3)),
        },
        "perbandingan_kalibrasi": hasil_cal,
        "kalibrator_terpilih": terbaik_nama,
        "parameter_final_3fold": final_params,
        "threshold_f1_dari_f12": best_t_f1,
        "pengali_bayes_empiris_f12": best_k,
        "perbandingan_aturan_keputusan": decision_comp,
        "ringkasan_penghematan_f3": {
            "hemat_vs_tandai_nol_idr": float(c_none - c_bayes),
            "hemat_vs_tandai_nol_pct": float((1.0 - c_bayes / max(c_none, 1.0)) * 100.0),
            "hemat_vs_tandai_semua_idr": float(c_all - c_bayes),
            "hemat_vs_tandai_semua_pct": float((1.0 - c_bayes / max(c_all, 1.0)) * 100.0),
        },
    }
    if save_json_flag:
        save_json(hasil, RESULTS_DIR / "task2_calibration.json")
        print("[SAVED] task2_calibration.json & cache/task2_fold3_calibrated.npz")
    return hasil


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2f probability calibration and cost-based decision rule.")
    parser.add_argument("--no-save-json", action="store_true", help="Skip saving task2_calibration.json.")
    args = parser.parse_args()
    jalankan_kalibrasi_dan_keputusan(save_json_flag=not args.no_save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
