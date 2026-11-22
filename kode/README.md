# Panduan Reproduksi Kode — UTS Machine Learning 2026

Dokumen ini menjelaskan cara mereproduksi seluruh hasil audit data, validasi, pelatihan model, gambar/tabel laporan, dan ketiga file prediksi (`prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`) dalam satu perintah.

> *Status: Inisialisasi repositori. Perintah tunggal, daftar pustaka beserta versinya, dan random seed akan diperbarui seiring integrasi pipeline Tugas 1–3.*

## Prasyarat Data

Ekstrak dataset resmi UTS (`data_uts_ml.zip`) ke folder `data/` di root repositori (tidak ikut di-commit ke Git) sehingga struktur berikut tersedia:

```text
data/
├── kamus_data.md
├── regresi_train.csv.gz
├── regresi_test.csv.gz
├── klasifikasi_train.csv.gz
├── klasifikasi_test.csv.gz
└── clustering.csv.gz
```
