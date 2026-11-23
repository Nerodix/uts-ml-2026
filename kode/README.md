# Panduan Reproduksi Kode — UTS Machine Learning 2026

Seluruh eksperimen, audit data, pelatihan model, figur/tabel laporan, dan ketiga file prediksi (`prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`) dirancang agar dapat direproduksi secara deterministik dari satu perintah.

## Struktur Direktori `kode/`

```text
kode/
├── README.md                  # Panduan reproduksi ini
├── pyproject.toml             # Spesifikasi dependensi Python (PEP 621 / uv)
├── uv.lock                    # Lockfile dependensi persis
├── requirements.txt           # Versi paket terkunci untuk pip
├── run_all.py                 # Entrypoint perintah tunggal seluruh pipeline
├── src/
│   ├── config.py              # Path, konstanta domain, dan SEED global (= 42)
│   ├── utils.py               # Seeding, metrik UTS (pinball, coverage, cost), tema plot
│   ├── validate_outputs.py    # Validator kontrak kolom/baris/range file prediksi
│   ├── audit/                 # Audit data Tugas 1–3
│   ├── regresi/               # Pipeline Tugas 1 (estimasi durasi pengiriman)
│   ├── klasifikasi/           # Pipeline Tugas 2 (deteksi transaksi fraud)
│   └── clustering/            # Pipeline Tugas 3 (segmentasi pelanggan)
└── tests/
    └── test_core.py           # Verifikasi unit metrik evaluasi dan validator output
```

## Prasyarat Data

Ekstrak `data_uts_ml.zip` ke direktori `data/` di root repositori (tidak ikut di-commit ke Git) atau set variabel lingkungan `UTS_ML_DATA_DIR=/path/ke/data`:

```text
data/
├── kamus_data.md
├── regresi_train.csv.gz
├── regresi_test.csv.gz
├── klasifikasi_train.csv.gz
├── klasifikasi_test.csv.gz
└── clustering.csv.gz
```

## Instalasi Environment

### Opsi 1 — Menggunakan `uv` (direkomendasikan)

```bash
cd kode
uv sync --python 3.12
```

### Opsi 2 — Menggunakan `pip` + `venv`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r kode/requirements.txt
```

## Menjalankan Pipeline

### Verifikasi environment & ketersediaan dataset

```bash
python kode/run_all.py --check-env
python kode/tests/test_core.py
```

### Menjalankan seluruh pipeline (satu perintah)

```bash
python kode/run_all.py
```

Perintah ini menjalankan pipeline tiap tugas, lalu memvalidasi format berkas prediksi. Tahap yang belum diimplementasikan (`klasifikasi`, `clustering`) dilewati otomatis dan diintegrasikan lewat PR masing-masing; saat itu tambahkan juga nama tugasnya ke `TUGAS_SIAP` di `run_all.py`.

## Konvensi Grafik

Semua grafik dibuat pada ukuran cetaknya: `figsize=(LEBAR_GAMBAR, tinggi)` dengan `LEBAR_GAMBAR = 6,7` inci (sekitar 17 cm, selebar area teks), sehingga huruf 7 s.d. 8 pt di grafik tidak mengecil di laporan. Teks grafik juga mengikuti EYD lewat pembantu di `src/utils.py`: `configure_plotting()` (ukuran huruf seragam), `angka_id()` dan `label_batang()` (koma desimal pada angka), `sumbu_koma()` (koma desimal pada skala sumbu), `miring()` (istilah asing dicetak miring), dan `label_bulan()` (singkatan bulan Indonesia). Pakai pembantu yang sama untuk grafik Tugas 2 dan 3.

## Versi Pustaka & Random Seed

- **Python:** `3.12.13`
- **Global Seed:** `SEED = 42` (`PYTHONHASHSEED`, `random`, `numpy`, serta parameter `random_state`/`seed` pada seluruh model)
- **Pustaka utama (terkunci di `uv.lock` & `requirements.txt`):**

| Pustaka | Versi |
| --- | --- |
| `numpy` | `2.2.6` |
| `pandas` | `3.1.1` |
| `scipy` | `1.18.1` |
| `pyarrow` | `26.0.1` |
| `scikit-learn` | `1.9.1` |
| `lightgbm` | `4.7.0` |
| `xgboost` | `3.4.1` |
| `catboost` | `1.2.10` |
| `statsmodels` | `0.15.0` |
| `imbalanced-learn` | `0.14.2` |
| `shap` | `0.54.1` |
| `optuna` | `5.1.0` |
| `matplotlib` | `3.11.2` |
| `seaborn` | `0.13.2` |
| `joblib` | `1.6.0` |
