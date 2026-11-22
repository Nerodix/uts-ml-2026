"""Kalibrasi Ketidakpastian P90 dengan Conformal Prediction (Tugas 1e).

Mengimplementasikan tiga varian *one-sided Conformalized Quantile Regression*
(CQR) dengan koreksi kuantil sampel-terbatas ``ceil((n + 1) * (1 - alpha)) / n``:
1. ``AdditiveCQR``             : Pergeseran aditif global ``p90 + delta``.
2. ``MultiplicativeCQR``       : Skala multiplikatif global ``m * p90`` (adaptif
   terhadap heteroskedastisitas durasi pengiriman 15–400 jam).
3. ``GroupMultiplicativeCQR``  : *Mondrian / group-conditional CQR* per ``layanan``
   agar cakupan ~90% tercapai merata di tiap jenis layanan.

Evaluasi dilakukan secara kronologis dan jujur pada OOF Fold 3:
- **Kalibrasi:** Juli 2025 (34.436 paket)
- **Evaluasi:** Agustus 2025 (34.168 paket, murni masa depan)

Jalankan langsung:
    python kode/src/regresi/calibration.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    REGRESI_TRAIN_PATH,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.regresi.preprocess import DATE_COLUMN  # noqa: E402
from src.utils import (  # noqa: E402
    configure_plotting,
    p90_coverage,
    pinball_loss,
    save_json,
    set_global_seed,
)


def conformal_quantile(scores: np.ndarray, alpha: float = 0.10) -> float:
    """Hitung kuantil konformal sampel-terbatas ``ceil((n+1)*(1-alpha))/n``.

    Parameters
    ----------
    scores : np.ndarray
        Vektor skor ketidakcocokan (*nonconformity scores*) pada set kalibrasi.
    alpha : float, default=0.10
        Tingkat kesalahan target (``alpha = 0.10`` berarti target coverage 90%).

    Returns
    -------
    float
        Nilai kuantil konformal yang memberikan jaminan cakupan ``>= 1 - alpha``.
    """
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(arr)
    if n == 0:
        raise ValueError("Set kalibrasi tidak boleh kosong.")
    k = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha))))
    return float(arr[k - 1])


@dataclass
class AdditiveCQR:
    """Conformalized Quantile Regression aditif satu sisi (``p90 + delta``)."""

    alpha: float = 0.10
    delta_: float = 0.0

    def fit(self, y_cal: np.ndarray, p90_cal: np.ndarray) -> AdditiveCQR:
        """Pelajari koreksi aditif ``delta`` dari residual ``y - p90`` di set kalibrasi."""
        scores = np.asarray(y_cal, dtype=np.float64) - np.asarray(p90_cal, dtype=np.float64)
        self.delta_ = conformal_quantile(scores, alpha=self.alpha)
        return self

    def transform(self, p50: np.ndarray, p90: np.ndarray) -> np.ndarray:
        """Terapkan koreksi aditif sambil menjamin ``p90_cal >= p50``."""
        return np.maximum(np.asarray(p50, dtype=np.float64), np.asarray(p90, dtype=np.float64) + self.delta_)


@dataclass
class MultiplicativeCQR:
    """Conformalized Quantile Regression multiplikatif satu sisi (``factor * p90``).

    Cocok untuk target miring ke kanan dan heteroskedastik (durasi pengiriman
    pendek vs antar-pulau): faktor pengali otomatis menambah margin kecil pada
    rute cepat dan margin proporsional lebih besar pada rute volatil.
    """

    alpha: float = 0.10
    factor_: float = 1.0

    def fit(self, y_cal: np.ndarray, p90_cal: np.ndarray) -> MultiplicativeCQR:
        """Pelajari faktor pengali konformal dari rasio ``y / p90`` di set kalibrasi."""
        denom = np.maximum(np.asarray(p90_cal, dtype=np.float64), 1e-3)
        ratios = np.asarray(y_cal, dtype=np.float64) / denom
        self.factor_ = conformal_quantile(ratios, alpha=self.alpha)
        return self

    def transform(self, p50: np.ndarray, p90: np.ndarray) -> np.ndarray:
        """Terapkan faktor pengali sambil menjamin ``p90_cal >= p50``."""
        return np.maximum(np.asarray(p50, dtype=np.float64), np.asarray(p90, dtype=np.float64) * self.factor_)


@dataclass
class GroupMultiplicativeCQR:
    """Mondrian / Group-Conditional Multiplicative CQR per kategori (mis. ``layanan``).

    Menjamin cakupan P90 mendekati 90% di **setiap** kelompok layanan, bukan
    hanya pada rata-rata keseluruhan.
    """

    alpha: float = 0.10
    global_factor_: float = 1.0
    group_factors_: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        y_cal: np.ndarray,
        p90_cal: np.ndarray,
        groups_cal: pd.Series | np.ndarray,
    ) -> GroupMultiplicativeCQR:
        """Pelajari faktor pengali konformal untuk tiap grup (dengan fallback global)."""
        y_arr = np.asarray(y_cal, dtype=np.float64)
        denom = np.maximum(np.asarray(p90_cal, dtype=np.float64), 1e-3)
        ratios = y_arr / denom
        g_arr = np.asarray(groups_cal, dtype=str)

        self.global_factor_ = conformal_quantile(ratios, alpha=self.alpha)
        self.group_factors_ = {}
        for g in sorted(np.unique(g_arr).tolist()):
            mask = g_arr == g
            self.group_factors_[g] = conformal_quantile(ratios[mask], alpha=self.alpha)
        return self

    def transform(
        self,
        p50: np.ndarray,
        p90: np.ndarray,
        groups: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Terapkan faktor pengali spesifik-grup sambil menjamin ``p90_cal >= p50``."""
        g_ser = pd.Series(np.asarray(groups, dtype=str))
        factors = g_ser.map(self.group_factors_).fillna(self.global_factor_).to_numpy(dtype=np.float64)
        return np.maximum(np.asarray(p50, dtype=np.float64), np.asarray(p90, dtype=np.float64) * factors)


def _eval_p90(
    y_true: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    layanan: np.ndarray,
) -> dict[str, Any]:
    """Hitung Pinball P90, coverage global, lebar interval median, dan coverage per layanan."""
    per_layanan = {}
    for lay in sorted(np.unique(layanan).tolist()):
        m = layanan == lay
        per_layanan[lay] = {
            "pinball_p90": round(pinball_loss(y_true[m], p90[m], 0.9), 4),
            "p90_coverage": round(p90_coverage(y_true[m], p90[m]), 4),
        }
    return {
        "pinball_p90": round(pinball_loss(y_true, p90, 0.9), 4),
        "p90_coverage": round(p90_coverage(y_true, p90), 4),
        "abs_cov_gap_pct": round(abs(p90_coverage(y_true, p90) - 0.90) * 100.0, 2),
        "median_interval_width": round(float(np.median(p90 - p50)), 2),
        "per_layanan": per_layanan,
    }


def jalankan_eksperimen_kalibrasi(train: pd.DataFrame) -> dict[str, Any]:
    """Uji 3 metode CQR secara kronologis (Kalibrasi Jul 2025 -> Evaluasi Agu 2025).

    Memuat prediksi OOF Fold 3 dari ``kode/cache/task1_oof.npz`` (dihasilkan
    oleh ``models.py``), membagi secara ketat menurut tanggal (Juli = kalibrasi,
    Agustus = evaluasi masa depan), lalu melatih kalibrator final pada seluruh
    OOF Fold 3 (Jul–Agu) untuk dipakai pada prediksi test Sep–Okt.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih ``regresi_train.csv.gz``.

    Returns
    -------
    dict[str, Any]
        Hasil evaluasi Agustus (sebelum vs sesudah CQR), pemilihan metode terbaik,
        parameter kalibrasi final untuk Sep–Okt, dan nama file figur.
    """
    ensure_dirs()
    oof_path = CACHE_DIR / "task1_oof.npz"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"{oof_path} belum ada. Jalankan `python kode/src/regresi/models.py` terlebih dahulu."
        )

    oof = np.load(oof_path)
    idx_f3 = oof["idx_fold3"]
    y_f3 = oof["y_fold3"]
    df_f3 = train.iloc[idx_f3].reset_index(drop=True)
    dt_f3 = pd.to_datetime(df_f3[DATE_COLUMN])
    lay_f3 = df_f3["layanan"].astype(str).to_numpy()

    cal_mask = (dt_f3 < "2025-08-01").to_numpy()
    eval_mask = (dt_f3 >= "2025-08-01").to_numpy()

    per_model: dict[str, Any] = {}
    for model_name in ("lgbm", "catboost", "ensemble"):
        p50 = oof[f"{model_name}_p50_fold3"]
        p90 = oof[f"{model_name}_p90_fold3"]

        # Fit 3 metode CQR HANYA di Juli 2025
        add_cqr = AdditiveCQR(alpha=0.10).fit(y_f3[cal_mask], p90[cal_mask])
        mul_cqr = MultiplicativeCQR(alpha=0.10).fit(y_f3[cal_mask], p90[cal_mask])
        grp_cqr = GroupMultiplicativeCQR(alpha=0.10).fit(
            y_f3[cal_mask], p90[cal_mask], lay_f3[cal_mask]
        )

        # Evaluasi jujur di Agustus 2025
        y_ev = y_f3[eval_mask]
        p50_ev = p50[eval_mask]
        p90_ev = p90[eval_mask]
        lay_ev = lay_f3[eval_mask]

        p90_add = add_cqr.transform(p50_ev, p90_ev)
        p90_mul = mul_cqr.transform(p50_ev, p90_ev)
        p90_grp = grp_cqr.transform(p50_ev, p90_ev, lay_ev)

        per_model[model_name] = {
            "uncalibrated": _eval_p90(y_ev, p50_ev, p90_ev, lay_ev),
            "additive_cqr": {
                "delta_jul": round(add_cqr.delta_, 4),
                **_eval_p90(y_ev, p50_ev, p90_add, lay_ev),
            },
            "multiplicative_cqr": {
                "factor_jul": round(mul_cqr.factor_, 4),
                **_eval_p90(y_ev, p50_ev, p90_mul, lay_ev),
            },
            "group_multiplicative_cqr": {
                "factors_jul": {k: round(v, 4) for k, v in grp_cqr.group_factors_.items()},
                **_eval_p90(y_ev, p50_ev, p90_grp, lay_ev),
            },
        }

    # Pilih metode CQR terbaik pada Ensemble berdasarkan kombinasi Pinball P90 & kedekatan coverage ke 90%
    ens_res = per_model["ensemble"]
    candidates = ["additive_cqr", "multiplicative_cqr", "group_multiplicative_cqr"]
    best_method = min(
        candidates,
        key=lambda m: (ens_res[m]["pinball_p90"], ens_res[m]["abs_cov_gap_pct"]),
    )

    # Fit parameter kalibrasi FINAL pada seluruh Fold 3 OOF (Jul–Agu, n=68.604) untuk produksi Sep–Okt
    p50_ens = oof["ensemble_p50_fold3"]
    p90_ens = oof["ensemble_p90_fold3"]
    final_add = AdditiveCQR(alpha=0.10).fit(y_f3, p90_ens)
    final_mul = MultiplicativeCQR(alpha=0.10).fit(y_f3, p90_ens)
    final_grp = GroupMultiplicativeCQR(alpha=0.10).fit(y_f3, p90_ens, lay_f3)

    final_params = {
        "selected_method": best_method,
        "additive_delta_jul_agu": round(final_add.delta_, 4),
        "multiplicative_factor_jul_agu": round(final_mul.factor_, 4),
        "group_multiplicative_factors_jul_agu": {
            k: round(v, 4) for k, v in final_grp.group_factors_.items()
        },
        "n_calibration_rows": int(len(y_f3)),
    }

    # Simpan juga P90 terkalibrasi Fold 3 (memakai metode terpilih) untuk analisis 1f
    if best_method == "group_multiplicative_cqr":
        p90_f3_cal = final_grp.transform(p50_ens, p90_ens, lay_f3)
    elif best_method == "multiplicative_cqr":
        p90_f3_cal = final_mul.transform(p50_ens, p90_ens)
    else:
        p90_f3_cal = final_add.transform(p50_ens, p90_ens)

    np.savez_compressed(
        CACHE_DIR / "task1_fold3_calibrated.npz",
        idx_fold3=idx_f3,
        y_fold3=y_f3,
        p50_fold3=p50_ens,
        p90_raw_fold3=p90_ens,
        p90_cal_fold3=p90_f3_cal,
    )

    fig_path = plot_kalibrasi_conformal(per_model, best_method)

    result = {
        "task": "Tugas 1e — Kalibrasi Ketidakpastian P90 dengan Conformal Prediction (CQR)",
        "seed": SEED,
        "eval_protocol": {
            "calibration_period": f"2025-07-01..2025-07-31 (n={int(cal_mask.sum()):,})",
            "evaluation_period": f"2025-08-01..2025-08-31 (n={int(eval_mask.sum()):,})",
        },
        "per_model_aug_eval": per_model,
        "final_calibration_params": final_params,
        "figure": fig_path.name,
    }
    save_json(result, RESULTS_DIR / "task1_calibration.json")
    return result


def plot_kalibrasi_conformal(per_model: dict[str, Any], best_method: str) -> Path:
    """Buat grafik 2-panel dampak Conformal Prediction pada Agustus 2025."""
    ensure_dirs()
    configure_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))

    # Panel (a): Coverage sebelum vs sesudah CQR pada 3 model utama (Evaluasi Agustus)
    models = [("lgbm", "LightGBM"), ("catboost", "CatBoost"), ("ensemble", "Ensemble")]
    x = np.arange(len(models))
    w = 0.20
    methods = [
        ("uncalibrated", "Tanpa kalibrasi", "#dc2626"),
        ("additive_cqr", "Additive CQR", "#f59e0b"),
        ("multiplicative_cqr", "Multiplicative CQR", "#2563eb"),
        ("group_multiplicative_cqr", "Group-CQR (per layanan)", "#16a34a"),
    ]
    for i, (mkey, mlabel, color) in enumerate(methods):
        vals = [per_model[mk][mkey]["p90_coverage"] * 100.0 for mk, _ in models]
        offset = (i - 1.5) * w
        bars = axes[0].bar(x + offset, vals, width=w, color=color, label=mlabel, zorder=2)
        for bar in bars:
            h = bar.get_height()
            y_txt = h + 0.30 if 89.4 <= h <= 90.05 else h + 0.16
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, y_txt, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=6.9, zorder=4,
            )
    axes[0].axhline(90.0, color="#111827", linestyle="--", linewidth=1.2, label="Target 90%", zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in models], fontsize=9)
    axes[0].set_ylabel("P90 Coverage pada Agustus 2025 (%)")
    axes[0].set_title("(a) Kalibrasi Coverage P90 (Kalibrasi Jul → Uji Agu)")
    axes[0].set_ylim(84, 93.0)
    axes[0].legend(fontsize=7.6, loc="lower right")

    # Panel (b): Coverage kondisional per layanan pada Ensemble (Sebelum vs Sesudah CQR terpilih)
    ens = per_model["ensemble"]
    services = sorted(ens["uncalibrated"]["per_layanan"].keys())
    x2 = np.arange(len(services))
    w2 = 0.34
    before = [ens["uncalibrated"]["per_layanan"][s]["p90_coverage"] * 100.0 for s in services]
    after = [ens[best_method]["per_layanan"][s]["p90_coverage"] * 100.0 for s in services]
    pretty_name = {
        "additive_cqr": "Additive CQR",
        "multiplicative_cqr": "Multiplicative CQR",
        "group_multiplicative_cqr": "Group-CQR (per layanan)",
    }.get(best_method, best_method)

    b1 = axes[1].bar(x2 - w2 / 2, before, width=w2, color="#dc2626", label="Sebelum kalibrasi", zorder=2)
    b2 = axes[1].bar(x2 + w2 / 2, after, width=w2, color="#16a34a", label=f"Sesudah {pretty_name}", zorder=2)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            y_txt = h + 0.32 if 89.4 <= h <= 90.05 else h + 0.16
            axes[1].text(bar.get_x() + bar.get_width() / 2, y_txt, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, zorder=4)
    axes[1].axhline(90.0, color="#111827", linestyle="--", linewidth=1.2, label="Target 90%", zorder=3)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(services, fontsize=9)
    axes[1].set_ylabel("P90 Coverage per Layanan (%)")
    axes[1].set_title("(b) Coverage Kondisional per Layanan (Ensemble, Agu 2025)")
    axes[1].set_ylim(84, 93.0)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out_path = FIGURES_DIR / "task1_1e_conformal_calibration.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    """Jalankan eksperimen kalibrasi CQR Tugas 1e dari CLI."""
    set_global_seed(SEED)
    train = pd.read_csv(REGRESI_TRAIN_PATH)
    res = jalankan_eksperimen_kalibrasi(train)

    print("=== HASIL KALIBRASI CONFORMAL P90 (KALIBRASI JUL -> UJI AGU 2025) ===")
    for model_name in ("lgbm", "catboost", "ensemble"):
        print(f"\n[{model_name.upper()}]")
        for method in ("uncalibrated", "additive_cqr", "multiplicative_cqr", "group_multiplicative_cqr"):
            m = res["per_model_aug_eval"][model_name][method]
            print(
                f"  {method:26s} -> Pinball P90={m['pinball_p90']:.4f} | "
                f"cov90={m['p90_coverage']*100:.2f}% (gap={m['abs_cov_gap_pct']:.2f}%) | "
                f"lebar_med={m['median_interval_width']:.2f} jam"
            )
    print("\nMetode terpilih & parameter final (Jul–Agu):", res["final_calibration_params"])


if __name__ == "__main__":
    main()
