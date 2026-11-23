"""Audit Label dan Fitur Tugas 2a — Klasifikasi Deteksi Transaksi Fraud.

Dibangun modular dengan prinsip satu fungsi untuk satu pertanyaan audit:
- ``schema``  : Dimensi, kesesuaian kolom, duplikat ID/baris, irisan train–test.
- ``time``    : Rentang waktu, jumlah transaksi bulanan, dan status pengurutan baris.
- ``id``      : Keacakan ``id_transaksi``, ``id_pengguna``, ``id_merchant``.
- ``delay``   : Pembentukan label ``is_fraud`` & kurva *right-censoring* Agustus 2025.
- ``leakage`` : Bukti forensik *post-transaction leakage* pada ``flag_review_manual``.
- ``quality`` : Missingness terstruktur, anomali ``umur_merchant_hari < 0``,
                ``persen_saldo`` cap 50, entitas *unseen*, dan statistik kolom kunci.
- ``v_scan``  : Screening 50 fitur anonim ``v01–v50`` (8 sinyal vs 42 noise + drift ``v15``).
- ``shift``   : Pergeseran pola fraud bulanan (gelombang QRIS Agustus + 5 arketipe fraud).
- ``impact``  : 4 eksperimen dampak kuantitatif pada split kronologis matang.

Jalankan langsung:
    python kode/src/klasifikasi/audit.py --save-json
    python kode/src/klasifikasi/audit.py --checks delay leakage
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
from sklearn.metrics import average_precision_score, roc_auc_score

KODE_DIR = Path(__file__).resolve().parents[2]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    FRAUD_REVIEW_COST_IDR,
    KLASIFIKASI_TEST_PATH,
    KLASIFIKASI_TRAIN_PATH,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.klasifikasi.audit_dampak import jalankan_semua_dampak  # noqa: E402
from src.klasifikasi.preprocess import (  # noqa: E402
    ANONYMOUS_V_COLUMNS,
    CATEGORICAL_COLUMNS,
    DRIFT_NOISE_V_COLUMNS,
    ID_COLUMN,
    MATURE_TRAIN_END_DATE,
    MERCHANT_ID_COLUMN,
    NOISE_V_COLUMNS,
    SIGNAL_V_COLUMNS,
    SNAPSHOT_TIMESTAMP,
    TARGET_COLUMN,
    USER_ID_COLUMN,
    estimasi_kurva_maturitas_label,
    muat_data_klasifikasi,
)
from src.utils import (  # noqa: E402
    LEBAR_GAMBAR,
    angka_id,
    configure_plotting,
    label_batang,
    label_bulan,
    miring,
    save_json,
    set_global_seed,
    sumbu_koma,
    timer,
)

PILIHAN_CEK = (
    "schema",
    "time",
    "id",
    "delay",
    "leakage",
    "quality",
    "v_scan",
    "shift",
    "impact",
)


def cek_skema_dan_duplikat(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Periksa dimensi dataset, kesesuaian kolom, duplikat baris/ID, dan irisan ID.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih dari ``klasifikasi_train.csv.gz``.
    test : pd.DataFrame
        Data uji dari ``klasifikasi_test.csv.gz``.

    Returns
    -------
    dict[str, Any]
        Dimensi, selisih kolom, serta jumlah duplikat dan irisan ``id_transaksi``.
    """
    tr_cols = [c for c in train.columns if c != "waktu_dt"]
    te_cols = [c for c in test.columns if c != "waktu_dt"]
    return {
        "train_shape": [int(train.shape[0]), int(len(tr_cols))],
        "test_shape": [int(test.shape[0]), int(len(te_cols))],
        "train_only_cols": sorted(set(tr_cols) - set(te_cols)),
        "test_only_cols": sorted(set(te_cols) - set(tr_cols)),
        "dup_id_train": int(train[ID_COLUMN].duplicated().sum()),
        "dup_id_test": int(test[ID_COLUMN].duplicated().sum()),
        "dup_rows_train": int(train[tr_cols].duplicated().sum()),
        "dup_rows_test": int(test[te_cols].duplicated().sum()),
        "overlap_id_train_test": int(len(set(train[ID_COLUMN]) & set(test[ID_COLUMN]))),
    }


def cek_rentang_waktu_dan_urutan(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Periksa rentang tanggal transaksi, keterurutan baris, dan volume bulanan.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih klasifikasi dengan kolom ``waktu_dt``.
    test : pd.DataFrame
        Data uji klasifikasi dengan kolom ``waktu_dt``.

    Returns
    -------
    dict[str, Any]
        Batas waktu train/test, status urutan kronologis, serta jumlah baris per bulan.
    """
    tr_mon = train["waktu_dt"].dt.to_period("M").astype(str).value_counts().sort_index()
    te_mon = test["waktu_dt"].dt.to_period("M").astype(str).value_counts().sort_index()
    return {
        "train_min": str(train["waktu_dt"].min()),
        "train_max": str(train["waktu_dt"].max()),
        "test_min": str(test["waktu_dt"].min()),
        "test_max": str(test["waktu_dt"].max()),
        "train_terurut_waktu": bool(train["waktu_dt"].is_monotonic_increasing),
        "test_terurut_waktu": bool(test["waktu_dt"].is_monotonic_increasing),
        "volume_bulanan_train": {str(k): int(v) for k, v in tr_mon.items()},
        "volume_bulanan_test": {str(k): int(v) for k, v in te_mon.items()},
    }


def cek_keacakan_id(train: pd.DataFrame) -> dict[str, Any]:
    """Periksa korelasi nomor ID (``id_transaksi``, ``id_pengguna``, ``id_merchant``) terhadap waktu/target.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih klasifikasi.

    Returns
    -------
    dict[str, Any]
        Korelasi Spearman nomor ID terhadap stempel waktu dan label ``is_fraud``.
    """
    t_num = train[ID_COLUMN].str.slice(1).astype(np.int64)
    u_num = train[USER_ID_COLUMN].str.slice(1).astype(np.int64)
    m_num = train[MERCHANT_ID_COLUMN].str.slice(1).astype(np.int64)
    w_ns = train["waktu_dt"].astype("int64")
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)

    return {
        "spearman_id_transaksi_waktu": float(stats.spearmanr(t_num, w_ns).statistic),
        "spearman_id_transaksi_fraud": float(stats.spearmanr(t_num, y).statistic),
        "spearman_id_pengguna_fraud": float(stats.spearmanr(u_num, y).statistic),
        "spearman_id_merchant_fraud": float(stats.spearmanr(m_num, y).statistic),
        "catatan": "Nomor id_transaksi acak (tidak berkorelasi dengan waktu/label); "
        "seluruh ID entitas hanya digunakan untuk agregasi riwayat kausal, bukan sebagai angka mentah.",
    }


def cek_label_dan_delay_konfirmasi(
    train: pd.DataFrame,
    simpan_plot: bool = True,
) -> dict[str, Any]:
    """Audit pembentukan label ``is_fraud`` dan *right-censoring* menjelang snapshot 31 Agt 2025.

    Menghitung laju fraud bulanan (total, non-QRIS, QRIS), kurva maturitas per
    jarak hari ke snapshot, serta rincian harian 15–31 Agustus 2025 untuk
    membuktikan jeda konfirmasi fraud ~21–25 hari (dengan 0 fraud terkonfirmasi
    pada 29–31 Agustus 2025).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih klasifikasi dengan kolom ``waktu_dt``.
    simpan_plot : bool, default=True
        Bila ``True``, menyimpan figur ``task2_2a1_label_delay.png``.

    Returns
    -------
    dict[str, Any]
        Statistik prevalensi global, bulanan, kurva maturitas, dan 3 hari terakhir nol-fraud.
    """
    df = train.copy()
    df["bulan"] = df["waktu_dt"].dt.to_period("M").astype(str)
    df["tgl"] = df["waktu_dt"].dt.strftime("%Y-%m-%d")

    mature_mask = df["waktu_dt"] <= pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")
    bulanan_rows: list[dict[str, Any]] = []
    for bulan, grp in df.groupby("bulan"):
        nq = grp[grp["kanal"] != "qris"]
        qr = grp[grp["kanal"] == "qris"]
        bulanan_rows.append(
            {
                "bulan": str(bulan),
                "n": int(len(grp)),
                "fraud": int(grp[TARGET_COLUMN].sum()),
                "laju_pct": float(grp[TARGET_COLUMN].mean() * 100.0),
                "laju_non_qris_pct": float(nq[TARGET_COLUMN].mean() * 100.0),
                "laju_qris_pct": float(qr[TARGET_COLUMN].mean() * 100.0),
                "fraud_qris": int(qr[TARGET_COLUMN].sum()),
            }
        )

    kurva = estimasi_kurva_maturitas_label(df, SNAPSHOT_TIMESTAMP)

    # Rincian harian Agustus 2025
    aug = df[df["bulan"] == "2025-08"]
    harian_aug: list[dict[str, Any]] = []
    for tgl, grp in aug.groupby("tgl"):
        nq = grp[grp["kanal"] != "qris"]
        qr = grp[grp["kanal"] == "qris"]
        harian_aug.append(
            {
                "tanggal": str(tgl),
                "hari_agt": int(tgl[-2:]),
                "n": int(len(grp)),
                "fraud": int(grp[TARGET_COLUMN].sum()),
                "laju_pct": float(grp[TARGET_COLUMN].mean() * 100.0),
                "laju_non_qris_pct": float(nq[TARGET_COLUMN].mean() * 100.0),
                "laju_qris_pct": float(qr[TARGET_COLUMN].mean() * 100.0),
            }
        )

    tiga_hari_akhir = df[df["tgl"].isin(["2025-08-29", "2025-08-30", "2025-08-31"])]
    baseline_total_pct = float(df.loc[mature_mask, TARGET_COLUMN].mean() * 100.0)
    baseline_nq_pct = float(df.loc[mature_mask & (df["kanal"] != "qris"), TARGET_COLUMN].mean() * 100.0)

    # Ringkasan distribusi nominal fraud vs biaya review Rp 25.000 pada periode matang
    f_nom = df.loc[mature_mask & (df[TARGET_COLUMN] == 1), "nominal"]
    hasil = {
        "n_total": int(len(df)),
        "n_fraud_total": int(df[TARGET_COLUMN].sum()),
        "laju_fraud_total_pct": float(df[TARGET_COLUMN].mean() * 100.0),
        "n_matang_jan_jul": int(mature_mask.sum()),
        "n_fraud_matang_jan_jul": int(df.loc[mature_mask, TARGET_COLUMN].sum()),
        "laju_fraud_matang_pct": baseline_total_pct,
        "laju_fraud_matang_non_qris_pct": baseline_nq_pct,
        "laju_fraud_agustus_pct": float(aug[TARGET_COLUMN].mean() * 100.0),
        "tiga_hari_terakhir_n": int(len(tiga_hari_akhir)),
        "tiga_hari_terakhir_fraud": int(tiga_hari_akhir[TARGET_COLUMN].sum()),
        "nominal_fraud_matang": {
            "median_idr": float(f_nom.median()),
            "mean_idr": float(f_nom.mean()),
            "q10_idr": float(f_nom.quantile(0.10)),
            "q90_idr": float(f_nom.quantile(0.90)),
            "persen_le_25k": float((f_nom <= FRAUD_REVIEW_COST_IDR).mean() * 100.0),
            "total_kerugian_idr": float(f_nom.sum()),
        },
        "bulanan": bulanan_rows,
        "kurva_maturitas": kurva.to_dict(orient="records"),
        "harian_agustus": harian_aug,
    }

    if simpan_plot:
        _plot_label_delay(hasil, FIGURES_DIR / "task2_2a1_label_delay.png")
    return hasil


def _plot_label_delay(hasil: dict[str, Any], output_path: Path) -> Path:
    """Gambar figur 2 panel audit *right-censoring* & jeda konfirmasi label."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.7))

    # Panel A: Laju fraud bulanan total vs non-QRIS
    ax0 = axes[0]
    bln = hasil["bulanan"]
    x = np.arange(len(bln))
    labels = [label_bulan(r["bulan"]) for r in bln]
    y_tot = [r["laju_pct"] for r in bln]
    y_nq = [r["laju_non_qris_pct"] for r in bln]

    colors = ["#4c72b0"] * (len(bln) - 1) + ["#d62728"]
    bars = ax0.bar(x, y_tot, color=colors, width=0.62, label="Total semua kanal")
    ax0.plot(x, y_nq, color="#1b2a49", marker="o", linewidth=1.2, label="Non-QRIS")
    ax0.axhline(
        hasil["laju_fraud_matang_pct"],
        color="#2ca02c",
        linestyle="--",
        linewidth=0.9,
        label=f"Rata-rata Jan–Jul ({angka_id(hasil['laju_fraud_matang_pct'], 2)}%)",
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, max(y_nq) * 1.25)
    ax0.set_ylabel("Laju fraud terkonfirmasi (%)")
    ax0.set_title("(a) Laju fraud bulanan (Agt turun oleh sensor)")
    sumbu_koma(ax0, "y")
    label_batang(ax0, bars, desimal=2, ukuran=5.8)
    ax0.legend(loc="lower left", fontsize=6.0)

    # Panel B: Kurva maturitas non-QRIS berdasarkan jarak hari ke snapshot 31 Agt
    ax1 = axes[1]
    km = hasil["kurva_maturitas"]
    # Urutkan dari kiri (31–45h, matang) ke kanan (0–3h, mendekati 31 Agt)
    km_rev = list(reversed(km))
    xk = np.arange(len(km_rev))
    lbl_k = [r["rentang_hari"] for r in km_rev]
    mat_pct = [r["rasio_maturitas"] * 100.0 for r in km_rev]
    bar_cols = [
        "#2ca02c" if r["hari_min"] >= 25 else ("#ff7f0e" if r["hari_min"] >= 10 else "#d62728")
        for r in km_rev
    ]
    b_mat = ax1.bar(xk, mat_pct, color=bar_cols, width=0.65)
    ax1.axhline(100.0, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax1.set_xticks(xk)
    ax1.set_xticklabels(lbl_k, rotation=32, ha="right", fontsize=6.2)
    ax1.set_ylim(0, 122)
    ax1.set_xlabel(f"Jarak hari sebelum {miring('snapshot')} 31 Agt 2025")
    ax1.set_ylabel("Maturitas label non-QRIS (%)")
    ax1.set_title(f"(b) Kurva {miring('right')}-{miring('censoring')} konfirmasi label")
    sumbu_koma(ax1, "y")
    label_batang(ax1, b_mat, desimal=0, ukuran=5.8)

    fig.tight_layout(w_pad=1.4)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def cek_leakage_review_manual(
    train: pd.DataFrame,
    test: pd.DataFrame,
    simpan_plot: bool = True,
) -> dict[str, Any]:
    """Buktikan secara forensik bahwa ``flag_review_manual`` adalah *post-transaction leakage*.

    Membandingkan:
    1. Tabel silang ``flag_review_manual`` vs ``is_fraud`` pada data latih.
    2. Laju ``flag_review_manual = 1`` per bulan pada data latih (~1,71%) vs
       non-fraud latih (~1,20%) vs data uji Sep–Okt (~1,23%).
    3. Rasio rata-rata fitur risiko saat ``flag_review_manual = 1`` vs ``0`` di
       latih penuh (tinggi 2–9x), latih non-fraud (~1,0x), dan uji (~1,0x).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih klasifikasi.
    test : pd.DataFrame
        Data uji klasifikasi.
    simpan_plot : bool, default=True
        Bila ``True``, menyimpan figur ``task2_2a2_leakage_review.png``.

    Returns
    -------
    dict[str, Any]
        Statistik kontingensi, laju bulanan, dan rasio fitur risiko latih vs uji.
    """
    tr = train.copy()
    te = test.copy()
    tr["bulan"] = tr["waktu_dt"].dt.to_period("M").astype(str)
    te["bulan"] = te["waktu_dt"].dt.to_period("M").astype(str)

    ct = pd.crosstab(tr["flag_review_manual"], tr[TARGET_COLUMN])
    n_f1_y1 = int(ct.loc[1, 1])
    n_f1_y0 = int(ct.loc[1, 0])
    n_f0_y1 = int(ct.loc[0, 1])
    n_f0_y0 = int(ct.loc[0, 0])

    # Laju review bulanan (latih & uji)
    laju_bulanan = {
        str(k): float(v * 100.0)
        for k, v in tr.groupby("bulan")["flag_review_manual"].mean().items()
    }
    laju_bulanan.update(
        {
            str(k): float(v * 100.0)
            for k, v in te.groupby("bulan")["flag_review_manual"].mean().items()
        }
    )

    # Bukti independensi flag_review_manual terhadap risiko di data uji vs latih penuh
    fitur_cek = [
        ("trx_1j", "Trx 1 jam"),
        ("ganti_password_7h", "Ganti sandi"),
        ("gagal_login_24j", "Gagal masuk"),
        ("perangkat_baru", "Perangkat baru"),
        ("jarak_lokasi_km", "Jarak lokasi"),
        ("persen_saldo", "Persen saldo"),
        ("nominal", "Nominal"),
    ]
    tr_nf = tr[tr[TARGET_COLUMN] == 0]
    rasio_fitur: list[dict[str, Any]] = []
    for col, lbl in fitur_cek:
        m_tr1 = float(tr.loc[tr["flag_review_manual"] == 1, col].mean())
        m_tr0 = float(tr.loc[tr["flag_review_manual"] == 0, col].mean())
        m_nf1 = float(tr_nf.loc[tr_nf["flag_review_manual"] == 1, col].mean())
        m_nf0 = float(tr_nf.loc[tr_nf["flag_review_manual"] == 0, col].mean())
        m_te1 = float(te.loc[te["flag_review_manual"] == 1, col].mean())
        m_te0 = float(te.loc[te["flag_review_manual"] == 0, col].mean())
        rasio_fitur.append(
            {
                "kolom": col,
                "label": lbl,
                "rasio_latih_semua": float(m_tr1 / max(m_tr0, 1e-9)),
                "rasio_latih_non_fraud": float(m_nf1 / max(m_nf0, 1e-9)),
                "rasio_uji": float(m_te1 / max(m_te0, 1e-9)),
            }
        )

    mature = tr[tr["waktu_dt"] <= pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")]
    hasil = {
        "kontingensi_train": {
            "rev0_nonfraud": n_f0_y0,
            "rev0_fraud": n_f0_y1,
            "rev1_nonfraud": n_f1_y0,
            "rev1_fraud": n_f1_y1,
        },
        "proporsi_fraud_yang_direview_pct": float(n_f1_y1 / (n_f1_y1 + n_f0_y1) * 100.0),
        "precision_review_di_train_pct": float(n_f1_y1 / (n_f1_y1 + n_f1_y0) * 100.0),
        "laju_review_train_total_pct": float(tr["flag_review_manual"].mean() * 100.0),
        "laju_review_train_nonfraud_pct": float(tr_nf["flag_review_manual"].mean() * 100.0),
        "laju_review_test_total_pct": float(te["flag_review_manual"].mean() * 100.0),
        "roc_auc_univariat_matang": float(roc_auc_score(mature[TARGET_COLUMN], mature["flag_review_manual"])),
        "pr_auc_univariat_matang": float(average_precision_score(mature[TARGET_COLUMN], mature["flag_review_manual"])),
        "laju_review_bulanan_pct": laju_bulanan,
        "rasio_fitur_risiko_rev1_vs_rev0": rasio_fitur,
    }

    if simpan_plot:
        _plot_leakage_review(hasil, FIGURES_DIR / "task2_2a2_leakage_review.png")
    return hasil


def _plot_leakage_review(hasil: dict[str, Any], output_path: Path) -> Path:
    """Gambar figur 2 panel pembuktian forensik leakage ``flag_review_manual``."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.7))

    # Panel A: Laju bulanan flag_review_manual=1 di Latih vs Uji
    ax0 = axes[0]
    bulanan = hasil["laju_review_bulanan_pct"]
    kunci = sorted(bulanan.keys())
    vals = [bulanan[k] for k in kunci]
    lbls = [label_bulan(k) for k in kunci]
    cols = ["#4c72b0" if k <= "2025-08" else "#d62728" for k in kunci]
    bars = ax0.bar(np.arange(len(kunci)), vals, color=cols, width=0.64)
    ax0.axhline(
        hasil["laju_review_train_nonfraud_pct"],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.0,
        label=f"Laju acak non-fraud ({angka_id(hasil['laju_review_train_nonfraud_pct'], 2)}%)",
    )
    ax0.axvline(7.5, color="#333333", linestyle=":", linewidth=0.9)
    ax0.set_xticks(np.arange(len(kunci)))
    ax0.set_xticklabels(lbls, fontsize=6.4)
    ax0.set_ylim(0, max(vals) * 1.28)
    ax0.set_ylabel("Proporsi flag_review_manual = 1 (%)")
    ax0.set_title("(a) Laju tinjauan manual: latih vs uji")
    sumbu_koma(ax0, "y")
    label_batang(ax0, bars, desimal=2, ukuran=5.4, putar=True)
    ax0.legend(loc="lower left", fontsize=6.0)

    # Panel B: Rasio nilai fitur risiko (rev=1 / rev=0) Latih Semua vs Latih Non-Fraud vs Uji
    ax1 = axes[1]
    rf = hasil["rasio_fitur_risiko_rev1_vs_rev0"]
    y_pos = np.arange(len(rf))
    h = 0.26
    r_tr = [r["rasio_latih_semua"] for r in rf]
    r_nf = [r["rasio_latih_non_fraud"] for r in rf]
    r_te = [r["rasio_uji"] for r in rf]

    ax1.barh(y_pos + h, r_tr, height=h, color="#d62728", label="Latih (semua)")
    ax1.barh(y_pos, r_nf, height=h, color="#7f7f7f", label="Latih (non-fraud)")
    ax1.barh(y_pos - h, r_te, height=h, color="#1f77b4", label="Uji (Sep–Okt)")
    ax1.axvline(1.0, color="#333333", linestyle="--", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r["label"] for r in rf], fontsize=6.6)
    ax1.invert_yaxis()
    ax1.set_xlabel("Rasio rata-rata (review = 1 / review = 0)")
    ax1.set_title("(b) Sinyal risiko di balik flag_review_manual")
    sumbu_koma(ax1, "x")
    ax1.legend(loc="lower right", fontsize=6.0)

    fig.tight_layout(w_pad=1.5)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def cek_kualitas_fitur_dan_anomali(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Audit nilai kosong, anomali rentang numerik, kardinalitas/unseen, & statistik kolom kunci.

    Parameters
    ----------
    train : pd.DataFrame
        Data latih klasifikasi.
    test : pd.DataFrame
        Data uji klasifikasi.

    Returns
    -------
    dict[str, Any]
        Ringkasan nilai kosong, anomali ``umur_merchant_hari < 0``, cap ``persen_saldo``,
        kardinalitas/unseen entitas & kategori, serta statistik kolom kunci.
    """
    mature = train[train["waktu_dt"] <= pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")]

    # 1. Missing values
    missing: dict[str, dict[str, float]] = {}
    for col in test.columns:
        if col == "waktu_dt":
            continue
        na_tr = float(train[col].isna().mean() * 100.0)
        na_te = float(test[col].isna().mean() * 100.0)
        if na_tr > 0 or na_te > 0:
            missing[col] = {"train_na_pct": na_tr, "test_na_pct": na_te}

    # Verifikasi missing menit_sejak_trx_terakhir == transaksi pertama pengguna di log latih
    tr_s = train.sort_values([USER_ID_COLUMN, "waktu_dt", ID_COLUMN], kind="mergesort")
    is_first = tr_s.groupby(USER_ID_COLUMN).cumcount() == 0
    kecocokan_first_tx = float((tr_s["menit_sejak_trx_terakhir"].isna().to_numpy() == is_first.to_numpy()).mean())

    # 2. Anomali umur_merchant_hari < 0
    neg_tr = train["umur_merchant_hari"] < 0
    neg_mat = mature["umur_merchant_hari"] < 0
    neg_te = test["umur_merchant_hari"] < 0

    # 3. Cap persen_saldo == 50
    cap_tr = train["persen_saldo"] >= 49.999
    cap_mat = mature["persen_saldo"] >= 49.999
    cap_te = test["persen_saldo"] >= 49.999

    # 4. Kardinalitas & unseen kategori + entitas
    entitas_cols = [USER_ID_COLUMN, MERCHANT_ID_COLUMN] + list(CATEGORICAL_COLUMNS)
    kardinalitas: dict[str, dict[str, Any]] = {}
    for col in entitas_cols:
        tr_u = set(train[col].dropna().unique())
        te_u = set(test[col].dropna().unique())
        unseen = te_u - tr_u
        kardinalitas[col] = {
            "train_unik": int(len(tr_u)),
            "test_unik": int(len(te_u)),
            "unseen_unik": int(len(unseen)),
            "unseen_baris_test_pct": float(test[col].isin(unseen).mean() * 100.0),
        }

    # 5. Repeat fraud pengguna & merchant (dasar motivasi fitur perilaku 2c)
    u_f = mature.groupby(USER_ID_COLUMN)[TARGET_COLUMN].sum()
    m_f = mature.groupby(MERCHANT_ID_COLUMN)[TARGET_COLUMN].sum()

    # 6. Statistik ringkas kolom numerik kunci (untuk tabel laporan)
    kolom_kunci = [
        "flag_review_manual",
        "nominal",
        "menit_sejak_trx_terakhir",
        "trx_1j",
        "jarak_lokasi_km",
        "umur_perangkat_hari",
        "umur_akun_hari",
        "persen_saldo",
        "umur_merchant_hari",
        "refund_merchant_30h",
        "v03",
        "v15",
        "v26",
        "v29",
        "v38",
    ]
    stat_kunci: list[dict[str, Any]] = []
    for col in kolom_kunci:
        s_tr = train[col].dropna().to_numpy(dtype=np.float64)
        s_te = test[col].dropna().to_numpy(dtype=np.float64)
        valid_m = mature[col].notna().to_numpy()
        auc_m = float(roc_auc_score(mature.loc[valid_m, TARGET_COLUMN], mature.loc[valid_m, col]))
        stat_kunci.append(
            {
                "kolom": col,
                "roc_auc_matang": auc_m,
                "ks_train_vs_test": float(stats.ks_2samp(s_tr, s_te).statistic),
                "train_min": float(np.min(s_tr)),
                "train_median": float(np.median(s_tr)),
                "train_max": float(np.max(s_tr)),
                "test_median": float(np.median(s_te)),
            }
        )

    return {
        "missing": missing,
        "kecocokan_missing_dengan_trx_pertama": kecocokan_first_tx,
        "laju_fraud_trx_pertama_matang_pct": float(
            mature.loc[mature["menit_sejak_trx_terakhir"].isna(), TARGET_COLUMN].mean() * 100.0
        ),
        "laju_fraud_trx_berulang_matang_pct": float(
            mature.loc[mature["menit_sejak_trx_terakhir"].notna(), TARGET_COLUMN].mean() * 100.0
        ),
        "umur_merchant_negatif": {
            "train_count": int(neg_tr.sum()),
            "train_pct": float(neg_tr.mean() * 100.0),
            "train_min_hari": float(train["umur_merchant_hari"].min()),
            "train_fraud_rate_pct": float(train.loc[neg_tr, TARGET_COLUMN].mean() * 100.0),
            "matang_fraud_rate_pct": float(mature.loc[neg_mat, TARGET_COLUMN].mean() * 100.0),
            "test_count": int(neg_te.sum()),
            "test_pct": float(neg_te.mean() * 100.0),
        },
        "persen_saldo_capped_50": {
            "train_count": int(cap_tr.sum()),
            "train_fraud_rate_pct": float(train.loc[cap_tr, TARGET_COLUMN].mean() * 100.0),
            "matang_fraud_rate_pct": float(mature.loc[cap_mat, TARGET_COLUMN].mean() * 100.0),
            "test_count": int(cap_te.sum()),
        },
        "kardinalitas_dan_unseen": kardinalitas,
        "pengulangan_fraud_matang": {
            "pengguna_ge1_fraud": int((u_f >= 1).sum()),
            "pengguna_ge2_fraud": int((u_f >= 2).sum()),
            "maks_fraud_per_pengguna": int(u_f.max()),
            "merchant_ge1_fraud": int((m_f >= 1).sum()),
            "merchant_ge2_fraud": int((m_f >= 2).sum()),
            "maks_fraud_per_merchant": int(m_f.max()),
        },
        "statistik_kolom_kunci": stat_kunci,
    }


def cek_screening_fitur_v(
    train: pd.DataFrame,
    test: pd.DataFrame,
    simpan_plot: bool = True,
) -> dict[str, Any]:
    """Screening seluruh 50 fitur anonim ``v01–v50`` berdasarkan sinyal fraud & *drift* waktu.

    Memisahkan 8 fitur sinyal nyata (``v03, v11, v19, v26, v29, v38, v42, v47``)
    dari 42 fitur noise, sekaligus mengidentifikasi ``v15`` sebagai fitur noise
    yang mengalami *drift* linear ekstrem terhadap waktu (KS = 0,502, AUC = 0,503).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih klasifikasi.
    test : pd.DataFrame
        Data uji klasifikasi.
    simpan_plot : bool, default=True
        Bila ``True``, menyimpan figur ``task2_2a3_v_screening.png``.

    Returns
    -------
    dict[str, Any]
        Daftar klasifikasi 50 fitur ``v``, statistik per kolom, dan ringkasan ``v15``.
    """
    mature = train[train["waktu_dt"] <= pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")]
    aug = train[train["waktu_dt"] > pd.Timestamp(f"{MATURE_TRAIN_END_DATE} 23:59:59")]
    y_mat = mature[TARGET_COLUMN].to_numpy(dtype=np.int8)
    y_aug = aug[TARGET_COLUMN].to_numpy(dtype=np.int8)
    w_tr = train["waktu_dt"].astype("int64").to_numpy()

    rows: list[dict[str, Any]] = []
    for col in ANONYMOUS_V_COLUMNS:
        auc_mat = float(roc_auc_score(y_mat, mature[col]))
        auc_aug = float(roc_auc_score(y_aug, aug[col]))
        ks = float(stats.ks_2samp(train[col].to_numpy(), test[col].to_numpy()).statistic)
        r_time = float(stats.spearmanr(w_tr, train[col].to_numpy()).statistic)
        if col in SIGNAL_V_COLUMNS:
            status = "sinyal"
        elif col in DRIFT_NOISE_V_COLUMNS:
            status = "drift_noise"
        else:
            status = "noise"
        rows.append(
            {
                "kolom": col,
                "status": status,
                "auc_matang": auc_mat,
                "kekuatan_sinyal_matang": float(abs(auc_mat - 0.5)),
                "auc_agustus": auc_aug,
                "kekuatan_sinyal_agustus": float(abs(auc_aug - 0.5)),
                "ks_train_test": ks,
                "spearman_waktu_train": r_time,
            }
        )

    df_v = pd.DataFrame(rows).sort_values("kekuatan_sinyal_matang", ascending=False)
    v15_row = next(r for r in rows if r["kolom"] == "v15")

    hasil = {
        "signal_v_cols": list(SIGNAL_V_COLUMNS),
        "drift_noise_v_cols": list(DRIFT_NOISE_V_COLUMNS),
        "n_signal": len(SIGNAL_V_COLUMNS),
        "n_noise": len(NOISE_V_COLUMNS),
        "v15_detail": v15_row,
        "semua_v": df_v.to_dict(orient="records"),
    }

    if simpan_plot:
        _plot_v_screening(df_v, FIGURES_DIR / "task2_2a3_v_screening.png")
    return hasil


def _plot_v_screening(df_v: pd.DataFrame, output_path: Path) -> Path:
    """Gambar peta screening 50 fitur anonim ``v01–v50`` (KS drift vs kekuatan sinyal)."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.65), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel A: Scatter KS (latih vs uji) vs |AUC - 0,5| (Jan–Jul matang)
    ax0 = axes[0]
    for status, warna, marker, lbl in [
        ("noise", "#b0b0b0", "o", "Noise (41 fitur)"),
        ("sinyal", "#1f77b4", "s", "Sinyal dipakai (8 fitur)"),
        ("drift_noise", "#d62728", "^", "Noise drift waktu (v15)"),
    ]:
        sub = df_v[df_v["status"] == status]
        ax0.scatter(
            sub["ks_train_test"],
            sub["kekuatan_sinyal_matang"],
            c=warna,
            marker=marker,
            s=28 if status != "noise" else 16,
            alpha=0.9 if status != "noise" else 0.6,
            label=lbl,
            zorder=3 if status != "noise" else 2,
        )

    # Anotasi fitur penting dengan offset rapi agar tidak tumpang tindih
    offsets = {
        "v03": (5, -2),
        "v38": (5, -2),
        "v11": (5, 4),
        "v42": (19, -2),
        "v19": (5, -7),
        "v29": (5, 2),
        "v26": (5, -2),
        "v47": (5, 3),
        "v15": (-18, 7),
    }
    for _, row in df_v[df_v["kolom"].isin(offsets)].iterrows():
        dx, dy = offsets[row["kolom"]]
        ax0.annotate(
            row["kolom"],
            (row["ks_train_test"], row["kekuatan_sinyal_matang"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.2,
            fontweight="bold" if row["kolom"] == "v15" else "normal",
            color="#d62728" if row["kolom"] == "v15" else "#1f2937",
        )
    ax0.set_xlabel("Pergeseran distribusi KS (latih vs uji)")
    ax0.set_ylabel(f"Kekuatan sinyal |ROC-{miring('AUC')} \u2212 0,5|")
    ax0.set_title("(a) Peta sinyal vs pergeseran 50 fitur v")
    sumbu_koma(ax0, "xy")
    ax0.legend(loc="center right", fontsize=6.0)

    # Panel B: Pergeseran sinyal 8 fitur v (Jan–Jul matang vs Agustus)
    ax1 = axes[1]
    sig_df = df_v[df_v["status"] == "sinyal"].sort_values("kekuatan_sinyal_matang", ascending=True)
    y_p = np.arange(len(sig_df))
    h = 0.36
    ax1.barh(y_p + h / 2, sig_df["kekuatan_sinyal_matang"], height=h, color="#1f77b4", label="Jan–Jul (matang)")
    ax1.barh(y_p - h / 2, sig_df["kekuatan_sinyal_agustus"], height=h, color="#ff7f0e", label="Agustus (shift)")
    ax1.set_yticks(y_p)
    ax1.set_yticklabels(sig_df["kolom"].tolist())
    ax1.set_xlabel(f"|ROC-{miring('AUC')} \u2212 0,5|")
    ax1.set_title("(b) Pergeseran sinyal 8 fitur v (v26 naik)")
    sumbu_koma(ax1, "x")
    ax1.legend(loc="lower right", fontsize=6.0)

    fig.tight_layout(w_pad=1.5)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def cek_pergeseran_pola_fraud(
    train: pd.DataFrame,
    test: pd.DataFrame,
    simpan_plot: bool = True,
) -> dict[str, Any]:
    """Audit pergeseran pola fraud lintas waktu dan identifikasi 5 arketipe modus fraud.

    Mengukur secara kuantitatif:
    - Lonjakan 8x laju fraud kanal ``qris`` pada Agustus 2025 (0,119% -> 0,789% meski
      tersensor; 1,27% pada Agt 1–6) dan lonjakan 30x pada ``jasa_lainnya``.
    - Ciri khas nominal bulat kelipatan Rp 100.000 pada gelombang QRIS (79,3% fraud QRIS
      Agustus vs 0,1% transaksi normal; berlanjut pada 461 transaksi uji Sep–Okt).
    - Distribusi 5 arketipe modus fraud (rule-based transparan dari temuan klasterisasi):
      1. *QRIS Bulat / Gelombang Baru*
      2. *Collusive Merchant Cashout* (nominal besar >= Rp 1,5 jt & ``refund_merchant_30h`` tinggi / ``persen_saldo`` tinggi)
      3. *Card Testing / Velocity Burst* (nominal kecil <= Rp 100 rb & ``trx_1j >= 2`` atau ``jumlah_akun_per_perangkat >= 2``)
      4. *Account Takeover (ATO)* (``perangkat_baru`` / ``ganti_password_7h`` / ``gagal_login_24j`` / ``jarak_lokasi_km`` jauh)
      5. *Merchant Anomali / Lainnya* (termasuk ``umur_merchant_hari < 0``).

    Parameters
    ----------
    train : pd.DataFrame
        Data latih klasifikasi.
    test : pd.DataFrame
        Data uji klasifikasi.
    simpan_plot : bool, default=True
        Bila ``True``, menyimpan figur ``task2_2a4_pattern_shift.png``.

    Returns
    -------
    dict[str, Any]
        Laju fraud per kanal bulanan, bukti kelanjutan QRIS bulat di data uji,
        dan ringkasan 5 arketipe modus fraud.
    """
    tr = train.copy()
    te = test.copy()
    tr["bulan"] = tr["waktu_dt"].dt.to_period("M").astype(str)
    te["bulan"] = te["waktu_dt"].dt.to_period("M").astype(str)

    # 1. Laju fraud per kanal per bulan
    piv_kanal = tr.pivot_table(index="kanal", columns="bulan", values=TARGET_COLUMN, aggfunc="mean") * 100.0
    laju_kanal_bulanan = {
        str(k): {str(b): float(v) for b, v in row.items()}
        for k, row in piv_kanal.iterrows()
    }

    # 2. Tren QRIS nominal bulat kelipatan Rp 100.000 lintas 10 bulan (Jan–Okt)
    tr_q = tr[tr["kanal"] == "qris"].copy()
    te_q = te[te["kanal"] == "qris"].copy()
    tr_q["bulat100k"] = (tr_q["nominal"] % 100_000 == 0).astype(np.int8)
    te_q["bulat100k"] = (te_q["nominal"] % 100_000 == 0).astype(np.int8)

    qris_bulat_bulanan: list[dict[str, Any]] = []
    for bln, grp in tr_q.groupby("bulan"):
        sub_b = grp[grp["bulat100k"] == 1]
        qris_bulat_bulanan.append(
            {
                "bulan": str(bln),
                "split": "train",
                "n_qris": int(len(grp)),
                "n_bulat100k": int(len(sub_b)),
                "persen_bulat100k": float(len(sub_b) / len(grp) * 100.0),
                "fraud_bulat100k": int(sub_b[TARGET_COLUMN].sum()),
                "laju_fraud_bulat100k_pct": float(sub_b[TARGET_COLUMN].mean() * 100.0) if len(sub_b) else 0.0,
            }
        )
    for bln, grp in te_q.groupby("bulan"):
        sub_b = grp[grp["bulat100k"] == 1]
        qris_bulat_bulanan.append(
            {
                "bulan": str(bln),
                "split": "test",
                "n_qris": int(len(grp)),
                "n_bulat100k": int(len(sub_b)),
                "persen_bulat100k": float(len(sub_b) / len(grp) * 100.0),
                "fraud_bulat100k": None,
                "laju_fraud_bulat100k_pct": None,
            }
        )

    # 3. Kategori merchant dengan lonjakan terbesar di Agustus (jasa_lainnya)
    mat_tr = tr[tr["bulan"] < "2025-08"]
    aug_tr = tr[tr["bulan"] == "2025-08"]
    jasa_jj = float(mat_tr.loc[mat_tr["kategori_merchant"] == "jasa_lainnya", TARGET_COLUMN].mean() * 100.0)
    jasa_aug = float(aug_tr.loc[aug_tr["kategori_merchant"] == "jasa_lainnya", TARGET_COLUMN].mean() * 100.0)

    # 4. Klasifikasi 5 arketipe modus fraud pada seluruh 2.622 transaksi fraud latih
    fraud = tr[tr[TARGET_COLUMN] == 1].copy()
    fraud["arketipe"] = tetapkan_arketipe_fraud(fraud)
    ringkasan_arketipe: list[dict[str, Any]] = []
    for nama, grp in fraud.groupby("arketipe"):
        ringkasan_arketipe.append(
            {
                "arketipe": str(nama),
                "n_fraud": int(len(grp)),
                "pangsa_jumlah_pct": float(len(grp) / len(fraud) * 100.0),
                "total_nominal_idr": float(grp["nominal"].sum()),
                "pangsa_kerugian_pct": float(grp["nominal"].sum() / fraud["nominal"].sum() * 100.0),
                "median_nominal_idr": float(grp["nominal"].median()),
                "n_jan_jul": int((grp["bulan"] < "2025-08").sum()),
                "n_agustus": int((grp["bulan"] == "2025-08").sum()),
            }
        )
    ringkasan_arketipe.sort(key=lambda r: r["total_nominal_idr"], reverse=True)

    aug_qris_f = aug_tr[(aug_tr["kanal"] == "qris") & (aug_tr[TARGET_COLUMN] == 1)]
    aug_qris_early_b = tr_q[
        (tr_q["waktu_dt"] >= pd.Timestamp("2025-08-01"))
        & (tr_q["waktu_dt"] <= pd.Timestamp("2025-08-10 23:59:59"))
        & (tr_q["bulat100k"] == 1)
    ]

    hasil = {
        "laju_qris_jan_jul_pct": float(mat_tr.loc[mat_tr["kanal"] == "qris", TARGET_COLUMN].mean() * 100.0),
        "laju_qris_agustus_pct": float(aug_tr.loc[aug_tr["kanal"] == "qris", TARGET_COLUMN].mean() * 100.0),
        "laju_jasa_lainnya_jan_jul_pct": jasa_jj,
        "laju_jasa_lainnya_agustus_pct": jasa_aug,
        "persen_bulat100k_pada_fraud_qris_agt": float((aug_qris_f["nominal"] % 100_000 == 0).mean() * 100.0),
        "persen_bulat100k_pada_normal_jan_jul": float(
            (mat_tr.loc[mat_tr[TARGET_COLUMN] == 0, "nominal"] % 100_000 == 0).mean() * 100.0
        ),
        "laju_fraud_qris_bulat100k_agt_awal_1_10_pct": float(aug_qris_early_b[TARGET_COLUMN].mean() * 100.0),
        "n_qris_bulat100k_test_sep_okt": int(te_q["bulat100k"].sum()),
        "laju_kanal_bulanan_pct": laju_kanal_bulanan,
        "qris_bulat100k_bulanan": qris_bulat_bulanan,
        "arketipe_fraud": ringkasan_arketipe,
    }

    if simpan_plot:
        _plot_pattern_shift(hasil, FIGURES_DIR / "task2_2a4_pattern_shift.png")
    return hasil


def tetapkan_arketipe_fraud(df: pd.DataFrame) -> pd.Series:
    """Petakan setiap transaksi (terutama kasus fraud) ke salah satu dari 5 modus utama.

    Aturan deterministik ini dipakai konsisten di audit 2a/2e dan pemecahan error 2g
    agar evaluasi per pola fraud mudah ditelusuri dan tidak bergantung pada label:
    1. ``Gelombang QRIS``   : transaksi pada kanal ``qris``.
    2. ``Merchant Kolusif`` : ``nominal >= 1,5 jt`` dan (``persen_saldo >= 5`` atau ``refund_merchant_30h >= 3,5``).
    3. ``Card Testing``     : ``nominal <= 150 rb`` dan (``trx_1j >= 1`` atau ``jumlah_akun_per_perangkat >= 2``).
    4. ``Pengambilalihan Akun`` : ``ganti_password_7h == 1`` atau ``gagal_login_24j >= 1``
                                  atau ``perangkat_baru == 1`` atau ``jarak_lokasi_km >= 100``.
    5. ``Anomali Merchant & Lainnya`` : sisanya (termasuk ``umur_merchant_hari < 0`` dan pola campuran).

    Parameters
    ----------
    df : pd.DataFrame
        Data transaksi yang memuat fitur domain dasar.

    Returns
    -------
    pd.Series
        Seri string berisi nama arketipe modus fraud per baris.
    """
    is_qris = df["kanal"] == "qris"
    is_collusive = (df["nominal"] >= 1_500_000) & (
        (df["persen_saldo"] >= 5.0) | (df["refund_merchant_30h"] >= 3.5)
    )
    is_card_test = (df["nominal"] <= 150_000) & (
        (df["trx_1j"] >= 1) | (df["jumlah_akun_per_perangkat"] >= 2)
    )
    is_ato = (
        (df["ganti_password_7h"] == 1)
        | (df["gagal_login_24j"] >= 1)
        | (df["perangkat_baru"] == 1)
        | (df["jarak_lokasi_km"] >= 100.0)
    )

    out = np.full(len(df), "Anomali Merchant & Lainnya", dtype=object)
    out[is_ato.to_numpy()] = "Pengambilalihan Akun (ATO)"
    out[is_card_test.to_numpy()] = "Card Testing & Velocity"
    out[is_collusive.to_numpy()] = "Merchant Kolusif (Nominal Besar)"
    out[is_qris.to_numpy()] = "Gelombang QRIS"
    return pd.Series(out, index=df.index, name="arketipe")


def _plot_pattern_shift(hasil: dict[str, Any], output_path: Path) -> Path:
    """Gambar figur 2 panel pergeseran pola fraud QRIS Agustus & kelanjutannya di uji."""
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(LEBAR_GAMBAR, 2.7))

    # Panel A: Laju fraud bulanan per kanal (Jan–Agt)
    ax0 = axes[0]
    k_map = hasil["laju_kanal_bulanan_pct"]
    bulan_list = sorted(next(iter(k_map.values())).keys())
    x0 = np.arange(len(bulan_list))
    gaya = {
        "qris": ("#d62728", "o", 2.0, "QRIS (naik 8× di Agt)"),
        "kartu": ("#1f77b4", "s", 1.1, "Kartu"),
        "web": ("#2ca02c", "^", 1.1, "Web"),
        "va": ("#9467bd", "d", 1.0, "VA"),
        "aplikasi": ("#7f7f7f", "v", 1.0, "Aplikasi"),
    }
    for kanal, (warna, mk, lw, lbl) in gaya.items():
        if kanal not in k_map:
            continue
        ys = [k_map[kanal][b] for b in bulan_list]
        ax0.plot(x0, ys, color=warna, marker=mk, linewidth=lw, label=lbl)
    ax0.set_xticks(x0)
    ax0.set_xticklabels([label_bulan(b) for b in bulan_list])
    ax0.set_ylim(0, 2.58)
    ax0.set_ylabel("Laju fraud teramati (%)")
    ax0.set_title("(a) Laju fraud bulanan per kanal")
    sumbu_koma(ax0, "y")
    ax0.legend(loc="upper center", ncol=2, fontsize=5.6)

    # Panel B: Jumlah transaksi QRIS nominal bulat kelipatan Rp 100 rb lintas Jan–Okt
    ax1 = axes[1]
    qb = hasil["qris_bulat100k_bulanan"]
    x1 = np.arange(len(qb))
    vals = [r["n_bulat100k"] for r in qb]
    cols = [
        "#a6cee3" if r["bulan"] < "2025-08" else ("#ff7f0e" if r["split"] == "train" else "#d62728")
        for r in qb
    ]
    bars = ax1.bar(x1, vals, color=cols, width=0.65)
    ax1.axvline(7.5, color="#333333", linestyle=":", linewidth=0.9)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([label_bulan(r["bulan"]) for r in qb], fontsize=6.4)
    ax1.set_ylim(0, max(vals) * 1.32)
    ax1.set_ylabel("Jumlah trx QRIS nominal bulat 100 rb")
    ax1.set_title("(b) Gelombang QRIS bulat berlanjut ke uji")
    sumbu_koma(ax1, "y")
    label_batang(ax1, bars, desimal=0, ukuran=5.8)
    ax1.annotate(
        "Uji (Sep–Okt)",
        xy=(8.5, max(vals) * 1.24),
        ha="center",
        va="top",
        fontsize=6.2,
        color="#d62728",
        fontweight="bold",
    )

    fig.tight_layout(w_pad=1.5)
    ensure_dirs()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def jalankan_audit_klasifikasi(
    cek_list: list[str] | None = None,
    simpan_json: bool = True,
) -> dict[str, Any]:
    """Jalankan seluruh atau sebagian langkah audit Tugas 2a.

    Parameters
    ----------
    cek_list : list[str] | None, default=None
        Daftar cek yang dijalankan; bila ``None``, menjalankan seluruh 9 cek.
    simpan_json : bool, default=True
        Bila ``True``, menyimpan ringkasan ke ``kode/results/task2_audit.json``.

    Returns
    -------
    dict[str, Any]
        Kamus hasil audit lengkap Tugas 2a.
    """
    set_global_seed(SEED)
    ensure_dirs()
    dipilih = list(PILIHAN_CEK) if not cek_list else list(cek_list)

    with timer("Muat dataset klasifikasi"):
        train, test = muat_data_klasifikasi(KLASIFIKASI_TRAIN_PATH, KLASIFIKASI_TEST_PATH)

    hasil: dict[str, Any] = {}
    if "schema" in dipilih:
        with timer("2a.1 Skema & duplikat"):
            hasil["skema"] = cek_skema_dan_duplikat(train, test)
            print(f"  Train={hasil['skema']['train_shape']}, Test={hasil['skema']['test_shape']}, dup_id={hasil['skema']['dup_id_train']}")
    if "time" in dipilih:
        with timer("2a.2 Rentang waktu & urutan"):
            hasil["waktu"] = cek_rentang_waktu_dan_urutan(train, test)
            print(f"  Train terurut={hasil['waktu']['train_terurut_waktu']}, Test terurut={hasil['waktu']['test_terurut_waktu']}")
    if "id" in dipilih:
        with timer("2a.3 Keacakan ID"):
            hasil["id"] = cek_keacakan_id(train)
            print(f"  Spearman(id_transaksi, waktu)={hasil['id']['spearman_id_transaksi_waktu']:.5f}")
    if "delay" in dipilih:
        with timer("2a.4 Label & kurva delay konfirmasi"):
            hasil["label_delay"] = cek_label_dan_delay_konfirmasi(train, simpan_plot=True)
            print(
                f"  Laju matang Jan–Jul={hasil['label_delay']['laju_fraud_matang_pct']:.3f}%, "
                f"Agt={hasil['label_delay']['laju_fraud_agustus_pct']:.3f}%, "
                f"29–31 Agt fraud={hasil['label_delay']['tiga_hari_terakhir_fraud']}/{hasil['label_delay']['tiga_hari_terakhir_n']}"
            )
    if "leakage" in dipilih:
        with timer("2a.5 Audit forensik leakage flag_review_manual"):
            hasil["leakage_review"] = cek_leakage_review_manual(train, test, simpan_plot=True)
            print(
                f"  Review rate train={hasil['leakage_review']['laju_review_train_total_pct']:.2f}% "
                f"(non-fraud={hasil['leakage_review']['laju_review_train_nonfraud_pct']:.2f}%), "
                f"test={hasil['leakage_review']['laju_review_test_total_pct']:.2f}%"
            )
    if "quality" in dipilih:
        with timer("2a.6 Kualitas fitur, missing, anomali, & unseen"):
            hasil["kualitas"] = cek_kualitas_fitur_dan_anomali(train, test)
            um = hasil["kualitas"]["umur_merchant_negatif"]
            print(
                f"  umur_merchant<0: n_tr={um['train_count']} (fraud={um['train_fraud_rate_pct']:.2f}%), "
                f"n_te={um['test_count']}"
            )
    if "v_scan" in dipilih:
        with timer("2a.7 Screening 50 fitur anonim v01–v50"):
            hasil["v_screening"] = cek_screening_fitur_v(train, test, simpan_plot=True)
            v15 = hasil["v_screening"]["v15_detail"]
            print(
                f"  8 sinyal v={hasil['v_screening']['signal_v_cols']}; "
                f"v15 drift KS={v15['ks_train_test']:.3f}, AUC={v15['auc_matang']:.3f}"
            )
    if "shift" in dipilih:
        with timer("2a.8 Pergeseran pola fraud & arketipe modus"):
            hasil["pattern_shift"] = cek_pergeseran_pola_fraud(train, test, simpan_plot=True)
            ps = hasil["pattern_shift"]
            print(
                f"  QRIS fraud Jan–Jul={ps['laju_qris_jan_jul_pct']:.3f}% -> Agt={ps['laju_qris_agustus_pct']:.3f}%; "
                f"QRIS bulat 100rb di Agt fraud={ps['persen_bulat100k_pada_fraud_qris_agt']:.1f}%, di test={ps['n_qris_bulat100k_test_sep_okt']}"
            )
    if "impact" in dipilih:
        with timer("2a.9 Eksperimen dampak kuantitatif (4 temuan)"):
            hasil["dampak"] = jalankan_semua_dampak(train, seed=SEED, simpan_plot=True)
            lk = hasil["dampak"]["leakage_review_manual"]
            print(
                f"  PR-AUC bocor CV={lk['dengan_review_cv_naif']['pr_auc']:.4f} -> "
                f"simulasi uji={lk['dengan_review_simulasi_uji']['pr_auc']:.4f} vs "
                f"jujur={lk['jujur_tanpa_review']['pr_auc']:.4f}"
            )

    if simpan_json:
        out_json = RESULTS_DIR / "task2_audit.json"
        save_json(hasil, out_json)
        print(f"[SAVED] {out_json.name}")
    return hasil


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2a label and feature audit (Fraud Detection).")
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=list(PILIHAN_CEK),
        default=None,
        help="Run a specific subset of audit checks (omit to run all).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save audit summary to kode/results/task2_audit.json.",
    )
    args = parser.parse_args()
    jalankan_audit_klasifikasi(cek_list=args.checks, simpan_json=args.save_json or args.checks is None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
