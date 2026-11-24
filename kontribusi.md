# Kontribusi Anggota Kelompok — UTS Machine Learning 2026

## Anggota

1. **Firdaus Arif Ramadhani** — NIM `101032300131` ([@fxrdhan](https://github.com/fxrdhan))
2. **Rainaldi Pakpahan** — NIM `101032300084` ([@rainaldipakpahan](https://github.com/rainaldipakpahan))

## Pembagian Peran & Daftar Pull Request

Setiap butir dikerjakan di *fork* masing-masing anggota dan digabungkan ke `main` lewat *pull request* (PR). Laporan Tugas 1 diunggah dalam satu PR, sedangkan bagian laporan Tugas 2 dan 3 diunggah per butir oleh anggota yang mengerjakan butir itu.

| Bagian | Firdaus Arif Ramadhani | Rainaldi Pakpahan |
| --- | --- | --- |
| Tugas 1 (Regresi) | 1c–1f | 1a–1b dan laporan Tugas 1 |
| Tugas 2 (Klasifikasi) | 2a–2b | 2c–2g |
| Tugas 3 (*Clustering*) | 3a–3b | 3c–3g |
| Pendukung | penyiapan repositori dan CI | perapian grafik dan integrasi `run_all.py` |

### Firdaus Arif Ramadhani (`@fxrdhan`)
- **Peran:** penyiapan repositori dan lingkungan yang dapat direproduksi (struktur `kode/`, utilitas bersama, CI); Tugas 1c–1f (model, kategori baru, kalibrasi konformal, analisis galat, dan `prediksi_regresi.csv`); Tugas 2a–2b (audit label dan fitur serta desain validasi); Tugas 3a–3b (prapemrosesan dan seleksi fitur); bagian laporan untuk butir 2a–2b dan 3a–3b.
- **Pull Requests:**
  - [#1](https://github.com/Nerodix/uts-ml-2026/pull/1): penyiapan lingkungan Python, utilitas bersama, dan pemeriksa format berkas prediksi
  - [#3](https://github.com/Nerodix/uts-ml-2026/pull/3): Tugas 1c–1f (model, kategori baru, kalibrasi CQR, SHAP, dan `prediksi_regresi.csv`)
  - [#5](https://github.com/Nerodix/uts-ml-2026/pull/5): Tugas 2a–2b (audit label dan kebocoran data serta validasi berbasis waktu)
  - [#7](https://github.com/Nerodix/uts-ml-2026/pull/7): CI GitHub Actions (pemeriksaan sintaks, *lint*, dan format CSV)
  - [#8](https://github.com/Nerodix/uts-ml-2026/pull/8): Tugas 3a–3b (prapemrosesan, seleksi fitur, dan pembobotan blok)
  - [#10](https://github.com/Nerodix/uts-ml-2026/pull/10): laporan Tugas 2a–2b
  - [#12](https://github.com/Nerodix/uts-ml-2026/pull/12): laporan Tugas 3a–3b

### Rainaldi Pakpahan (`@rainaldipakpahan`)
- **Peran:** *(akan dilengkapi sesuai PR)*.
- **Pull Requests:**
  - *(akan diisi)*

## Penggunaan Alat AI

Kelompok memakai dua alat AI sebagai asisten:

| Alat | Model | Bagian yang dibantu |
| --- | --- | --- |
| Antigravity | Gemini Neon | Pembuatan kode *pipeline* (audit, fitur, model, dan evaluasi) beserta grafik dan tabel hasilnya |
| Claude Code | Fable 5.2 | Penyusunan laporan (`laporan.pdf`) |

Pesan *commit* dan deskripsi PR juga ditulis dengan bantuan alat tersebut.

**Cara kelompok memverifikasi keluarannya:**
- Seluruh angka, grafik, dan berkas prediksi dihasilkan ulang dengan satu perintah (`python kode/run_all.py`) memakai *seed* tetap (42) dan versi pustaka yang dikunci di `kode/uv.lock` serta `kode/requirements.txt`.
- Setiap angka di laporan bersumber dari berkas hasil di `kode/results/`; skrip dan berkas asalnya dirinci per butir pada Lampiran A laporan.
- Format ketiga berkas prediksi diperiksa saat berkas ditulis dan oleh CI pada setiap PR.
- Setiap keputusan diuji dengan eksperimen pembanding (misalnya validasi berbasis waktu terhadap validasi acak, dengan dan tanpa kolom yang bocor, serta ablasi fitur), dan hasil yang tidak menguntungkan tetap dilaporkan.
- Perubahan masuk lewat PR per butir dengan satu langkah per *commit* sehingga dapat ditinjau anggota sebelum digabungkan ke `main`.

Kedua anggota bertanggung jawab atas seluruh isi kode dan laporan.

## Persetujuan Anggota

Seluruh anggota telah meninjau, memahami, dan menyetujui pembagian kontribusi di atas:
- [x] Firdaus Arif Ramadhani (`101032300131`)
- [ ] Rainaldi Pakpahan (`101032300084`)
