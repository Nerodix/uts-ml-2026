"""Utilitas reproduksibilitas, metrik evaluasi UTS, pencatatan hasil, dan tema plot."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import time
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from .config import SEED

TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "statsmodels",
    "imbalanced-learn",
    "shap",
    "optuna",
    "matplotlib",
    "seaborn",
    "joblib",
]


def atur_seed_global(seed: int = SEED) -> None:
    """Kunci random seed Python dan NumPy untuk reproduksibilitas."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def hitung_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Hitung quantile (pinball) loss pada kuantil q in (0, 1)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    diff = y_t - y_p
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def hitung_cakupan_p90(y_true: np.ndarray, y_pred_p90: np.ndarray) -> float:
    """Proporsi observasi aktual yang berada <= prediksi P90 (target 0.90)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred_p90, dtype=np.float64)
    return float(np.mean(y_t <= y_p))


def hitung_biaya_total_fraud(
    y_true: np.ndarray,
    flagged: np.ndarray,
    nominal: np.ndarray,
    review_cost: float = 25_000.0,
) -> float:
    """Total biaya klasifikasi fraud sesuai rumus UTS.

    cost = sum(nominal fraud yang lolos) + review_cost * jumlah transaksi ditandai.
    """
    y_t = np.asarray(y_true, dtype=np.int8)
    f = np.asarray(flagged, dtype=np.int8)
    nom = np.asarray(nominal, dtype=np.float64)
    missed_fraud_loss = float(np.sum(nom[(y_t == 1) & (f == 0)]))
    review_total = float(review_cost * np.sum(f == 1))
    return missed_fraud_loss + review_total


def tulis_json(data: Any, path: Path) -> None:
    """Simpan dictionary/list ke JSON dengan format rapi dan deterministik."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        fh.write("\n")


def baca_json(path: Path) -> Any:
    """Muat file JSON."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def kumpulkan_info_lingkungan() -> dict[str, Any]:
    """Kumpulkan versi Python, seed, dan pustaka ML utama untuk reproduksibilitas."""
    versions: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return {
        "python": platform.python_version(),
        "seed": SEED,
        "packages": versions,
    }


# Lebar gambar dalam inci (sekitar 17 cm, selebar area teks laporan A4). Karena gambar
# dibuat pada ukuran cetaknya, huruf 7 s.d. 8 pt di grafik tetap terbaca di laporan.
LEBAR_GAMBAR = 6.7
UKURAN_NILAI = 6.2  # ukuran huruf angka di atas batang
BULAN_SINGKAT = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agt", "Sep", "Okt", "Nov", "Des"]


def atur_gaya_grafik() -> None:
    """Set gaya grafik matplotlib/seaborn yang seragam untuk laporan.

    Ukuran huruf dipilih untuk gambar selebar ``LEBAR_GAMBAR`` inci, yaitu ukuran
    cetaknya di laporan, sehingga teks grafik tidak mengecil saat ditempel.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.titlepad": 4.0,
            "axes.labelsize": 7.5,
            "axes.labelpad": 2.5,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "legend.fontsize": 6.8,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.borderpad": 0.3,
            "legend.handlelength": 1.4,
            "legend.labelspacing": 0.3,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.2,
            "mathtext.default": "regular",
        }
    )


def angka_id(nilai: float, desimal: int = 2) -> str:
    """Format bilangan menurut EYD: koma sebagai tanda desimal dan titik sebagai pemisah ribuan.

    Parameters
    ----------
    nilai : float
        Bilangan yang akan ditulis.
    desimal : int, default=2
        Banyaknya angka di belakang koma.

    Returns
    -------
    str
        Misalnya ``angka_id(1234.5, 1)`` menghasilkan ``"1.234,5"``.
    """
    bulat = Decimal(str(nilai)).quantize(Decimal(1).scaleb(-desimal), rounding=ROUND_HALF_UP)
    teks = f"{bulat:,.{desimal}f}"
    return teks.replace(",", "\x00").replace(".", ",").replace("\x00", ".").replace("-", "\u2212")


def miring(teks: str) -> str:
    """Cetak miring istilah asing di dalam teks grafik (EYD), memakai mathtext matplotlib.

    Parameters
    ----------
    teks : str
        Istilah asing tanpa tanda hubung, misalnya ``"pinball loss"``.

    Returns
    -------
    str
        String mathtext yang dapat digabung dengan teks biasa pada judul, label, atau legenda.
    """
    return r"$\mathit{" + teks.replace(" ", r"\ ") + "}$"


def label_bulan(periode: str) -> str:
    """Ubah ``"2025-08"`` menjadi singkatan bulan Indonesia (``"Agt"``)."""
    return BULAN_SINGKAT[int(str(periode)[5:7]) - 1]


class _FormatKoma(ScalarFormatter):
    """Label skala sumbu dengan koma desimal dan titik ribuan; jumlah desimal seragam pada satu sumbu."""

    def __init__(self) -> None:
        super().__init__(useOffset=False, useMathText=False)
        self.set_scientific(False)

    def __call__(self, x: float, pos: int | None = None) -> str:
        teks = super().__call__(x, pos)
        if not teks:
            return teks
        tanda = "\u2212" if teks[0] in "-\u2212" else ""
        bulat, _, pecahan = teks.lstrip("-\u2212").partition(".")
        bulat = f"{int(bulat):,}".replace(",", ".")
        return tanda + bulat + ("," + pecahan if pecahan else "")


def sumbu_koma(ax: Any, sumbu: str = "y") -> None:
    """Tulis label skala sumbu numerik dengan koma desimal dan titik ribuan.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sumbu grafik yang diubah.
    sumbu : {"x", "y", "xy"}, default="y"
        Sumbu yang label skalanya diformat ulang.
    """
    if "x" in sumbu:
        ax.xaxis.set_major_formatter(_FormatKoma())
    if "y" in sumbu:
        ax.yaxis.set_major_formatter(_FormatKoma())


def label_batang(
    ax: Any,
    batang: Any,
    desimal: int = 2,
    putar: bool = False,
    mendatar: bool = False,
    di_dalam: bool = False,
    ukuran: float = UKURAN_NILAI,
) -> None:
    """Tulis nilai setiap batang di ujungnya dengan koma desimal.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sumbu grafik tempat batang digambar.
    batang : matplotlib.container.BarContainer
        Keluaran ``ax.bar`` atau ``ax.barh``.
    desimal : int, default=2
        Banyaknya angka di belakang koma.
    putar : bool, default=False
        Putar angka 90 derajat (untuk batang tegak yang rapat).
    mendatar : bool, default=False
        ``True`` untuk batang mendatar (``ax.barh``).
    di_dalam : bool, default=False
        Tulis angka berwarna putih di dalam ujung batang tegak (dipakai bila ada garis acuan
        di dekat puncak batang).
    ukuran : float
        Ukuran huruf angka.
    """
    for b in batang:
        if mendatar:
            nilai = b.get_width()
            ax.annotate(
                angka_id(nilai, desimal),
                (nilai, b.get_y() + b.get_height() / 2),
                xytext=(1.5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=ukuran,
            )
        else:
            nilai = b.get_height()
            ax.annotate(
                angka_id(nilai, desimal),
                (b.get_x() + b.get_width() / 2, nilai),
                xytext=(0, (-4.0 if putar else -7.0) if di_dalam else 1.5),
                textcoords="offset points",
                ha="center",
                va="top" if di_dalam else "bottom",
                rotation=90 if putar else 0,
                color="white" if di_dalam else None,
                fontsize=ukuran,
            )


@contextmanager
def catat_waktu(label: str) -> Iterator[None]:
    """Context manager untuk mencatat durasi eksekusi tiap tahap."""
    t0 = time.perf_counter()
    print(f"[START] {label}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[DONE]  {label} ({dt:.2f}s)")
