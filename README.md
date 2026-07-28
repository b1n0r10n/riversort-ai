# RiverSort AI

RiverSort AI adalah prototype *computer vision* untuk menganalisis gambar atau
video pendek permukaan sungai. Aplikasi mendeteksi sampah plastik mengambang,
menghitung komposisi visual, menyusun indeks prioritas pembersihan, dan
memberikan rekomendasi berbasis aturan yang dapat ditelusuri.

![Dashboard RiverSort AI](docs/assets/dashboard-upload.png)

## Kemampuan

- Analisis satu gambar JPEG/PNG atau video MP4 H.264.
- Deteksi empat kelas plastik dengan YOLO11n.
- Penghitungan objek unik pada video dengan ByteTrack dua tahap.
- Normalisasi otomatis untuk media beresolusi tinggi.
- Dua metrik komposisi: proporsi jumlah objek unik dan area deteksi.
- Indeks prioritas visual yang memperlihatkan komponen perhitungannya.
- Rekomendasi deterministik beserta nilai observasi dan ambang aturan.
- Preview beranotasi serta ekspor JSON dan CSV.
- Pemrosesan lokal tanpa layanan inferensi eksternal.

Model aktif mendukung:

```text
plastic_bottle
plastic_bag
plastic_packaging_sachet
other_plastic
```

Taksonomi target juga memuat `other_floating_debris`. Kelas tersebut belum
didukung model aktif dan dilaporkan sebagai `not_supported`, bukan `0%`.

## Instalasi

Prasyarat:

- Python 3.11;
- FFmpeg dan FFprobe;
- Git.

```bash
git clone https://github.com/b1n0r10n/riversort-ai.git
cd riversort-ai

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Unduh model terverifikasi:

```bash
python scripts/download_model.py
export RIVERSORT_MODEL_PATH="$PWD/models/riversort-yolo11n-fpwr4-0.1.0-baseline.1.pt"
```

Jalankan dashboard:

```bash
streamlit run app.py
```

Setelah model tersedia, aplikasi dapat digunakan tanpa koneksi internet.
Panduan lengkap terdapat di [docs/USAGE.md](docs/USAGE.md).

## Alur analisis

```mermaid
flowchart LR
    A[Unggah media] --> B[Validasi dan normalisasi]
    B --> C[Deteksi YOLO11n]
    C --> D{Jenis media}
    D -->|Gambar| E[Agregasi deteksi]
    D -->|Video| F[ByteTrack dua tahap]
    E --> G[Komposisi visual]
    F --> G
    G --> H[Indeks prioritas]
    H --> I[Rekomendasi berbasis aturan]
    I --> J[Preview dan ekspor]
```

Pada gambar, satu deteksi valid dihitung sebagai satu objek. Pada video,
deteksi antar-frame dihubungkan menjadi *track* sehingga satu objek yang sama
tidak dihitung kembali pada setiap frame. Kelas akhir sebuah track ditentukan
dengan *confidence-weighted vote* dan digunakan secara konsisten pada preview
serta hasil agregasi.

## Model dan data

Baseline menggunakan snapshot `riversort_fpwr4-20260722-r2` yang diturunkan
dari Float Plastic Waste River v6:

- 2.824 gambar dan 5.761 anotasi;
- split *group-aware* 1.947 train, 437 validation, dan 440 test;
- empat kelas plastik aktif;
- tidak memiliki *negative sample*;
- tidak memasukkan River Trash ke evaluasi numerik.

Model aktif adalah YOLO11n `0.1.0-baseline.1`. Pada held-out test FPWR dengan
confidence `0,40` dan IoU `0,50`, model memperoleh mAP50 `0,6936` dan macro-F1
`0,6904`. Nilai tersebut hanya menggambarkan evaluasi pada FPWR, bukan akurasi
nasional atau semua kondisi sungai.

Lihat [model card](models/MODEL_CARD.md) dan
[metodologi data](docs/DATA_AND_MODEL.md) untuk konteks lengkap.

## Struktur

```text
app.py                         entry point Streamlit
src/riversort/domain/          kontrak, metrik, dan aturan bisnis
src/riversort/application/     orkestrasi analisis gambar dan video
src/riversort/adapters/        YOLO, OpenCV, ByteTrack, FFmpeg, dan ekspor
src/riversort/infrastructure/  konfigurasi, logging, runtime, dan workspace
src/riversort/presentation/    dashboard Streamlit
configs/                       konfigurasi model, tracker, media, dan aturan
scripts/                       pipeline data/model serta utilitas rilis
tests/                         unit dan integration tests
docs/                          dokumentasi produk dan teknis
models/                        model card; weight diunduh terpisah
reports/                       ringkasan evaluasi dan snapshot
```

RiverSort AI menggunakan arsitektur *modular monolith*: satu aplikasi lokal
dengan batas dependensi yang jelas antara domain, orkestrasi, adapter, dan UI.

## Pengujian

```bash
python -m pytest -q
python -m pip check
python scripts/release/audit_public_repository.py
```

Test mencakup kontrak data, konfigurasi, validasi media, komposisi, prioritas,
rekomendasi, tracking, ekspor, cleanup workspace, dan smoke test dashboard.

## Batas penggunaan

RiverSort AI adalah alat bantu keputusan, bukan alat ukur fisik dan bukan
sistem pembersihan otomatis.

> **Estimasi komposisi visual berdasarkan jumlah objek unik dan proporsi area
> deteksi.**

Hasil tidak menyatakan berat, massa, tonase, atau volume fisik. Model dapat
menghasilkan *false positive* dan *false negative*, terutama pada objek kecil,
tertutup, atau adegan padat. Debris nonplastik belum didukung sehingga indeks
prioritas dapat lebih rendah daripada kondisi lapangan. Verifikasi manusia
tetap diperlukan sebelum tindakan pembersihan.

## Dokumentasi

| Dokumen | Isi |
|---|---|
| [PRODUCT.md](docs/PRODUCT.md) | Masalah, pengguna, tujuan, scope, dan requirement |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arsitektur, aliran data, tracking, serta aturan bisnis |
| [DATA_AND_MODEL.md](docs/DATA_AND_MODEL.md) | Dataset, snapshot, training, dan model |
| [EVALUATION.md](docs/EVALUATION.md) | Metrik, pengujian, failure modes, dan interpretasi |
| [USAGE.md](docs/USAGE.md) | Instalasi, konfigurasi, dashboard, dan ekspor |
| [ROADMAP.md](docs/ROADMAP.md) | Arah pengembangan dan validation gates |
| [DECISIONS.md](docs/DECISIONS.md) | Keputusan produk dan teknis yang aktif |

## Tim

- Bintang Akalla Junjunan — product, application, dan machine learning
- Nisa Setiawati — research, validation, dan communication

## Lisensi

Source code dan model RiverSort AI tersedia di bawah
[GNU Affero General Public License v3.0 or later](LICENSE). Dataset FPWR tetap
mengikuti lisensi CC BY 4.0 dari pemiliknya. Rincian atribusi terdapat di
[NOTICE.md](NOTICE.md).
