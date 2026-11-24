# UTS Machine Learning 2026 — Regresi, Klasifikasi, dan Clustering

Repositori kelompok untuk Ujian Tengah Semester (UTS) mata kuliah Machine Learning:
1. **Tugas 1 — Regresi:** Estimasi waktu pengiriman paket (`durasi_jam`, kuantil P50 dan P90).
2. **Tugas 2 — Klasifikasi:** Deteksi transaksi *fraud* (`is_fraud`, probabilitas terkalibrasi dan keputusan berbasis biaya).
3. **Tugas 3 — Clustering:** Segmentasi ~120.000 pelanggan beserta deteksi anomali (`-1`).

## Anggota Kelompok

| Nama | NIM | GitHub |
| --- | --- | --- |
| Firdaus Arif Ramadhani | 101032300131 | [@fxrdhan](https://github.com/fxrdhan) |
| Rainaldi Pakpahan | 101032300084 | [@rainaldipakpahan](https://github.com/rainaldipakpahan) |

## Struktur Repositori (`main`)

```text
.
├── laporan.pdf                 # Laporan lengkap Tugas 1–3 (≤18 halaman di luar lampiran)
├── kode/                       # Seluruh kode sumber, konfigurasi, dan panduan reproduksi
│   └── README.md               # Perintah tunggal untuk mereproduksi seluruh angka & output
├── prediksi_regresi.csv        # id_paket,p50,p90
├── prediksi_klasifikasi.csv    # id_transaksi,prob,tandai
├── cluster.csv                 # id_pelanggan,cluster
└── kontribusi.md               # Pembagian peran, tautan PR, dan deklarasi penggunaan AI
```

## Reproduksi

Lihat panduan lengkap, versi pustaka, *seed*, dan perintah tunggal reproduksi di [`kode/README.md`](kode/README.md).

## Diagram Alur

Diagram alur *pipeline* (satu alur umum, ringkasan tiap tugas, dan rincian tiap butir soal) tersedia di papan FigJam [Alur Umum Pipeline UTS ML](https://www.figma.com/board/cJ9pRoFQd8qn5P4u622eAd/Alur-Umum-Pipeline-UTS-ML). Diagram ini hanya pendamping; angka dan penjelasan lengkap tetap mengacu pada [`laporan.pdf`](laporan.pdf).
