# Usage guide

## 1. Prasyarat

- Linux atau sistem yang dapat menjalankan Python 3.11;
- FFmpeg dan FFprobe;
- ruang sekitar 1 GB untuk environment;
- model weight sekitar 5,3 MB;
- koneksi internet hanya untuk instalasi dan download awal.

Pada Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg python3.11 python3.11-venv
```

## 2. Instalasi

```bash
git clone https://github.com/b1n0r10n/riversort-ai.git
cd riversort-ai

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verifikasi:

```bash
python -m pip check
python -m pytest -q
```

## 3. Model

Unduh model dari release resmi:

```bash
python scripts/download_model.py
```

Script memverifikasi:

- SHA-256 weight;
- empat nama kelas;
- kecocokan dengan model version yang dipublikasikan.

Tetapkan path untuk sesi terminal:

```bash
export RIVERSORT_MODEL_PATH="$PWD/models/riversort-yolo11n-fpwr4-0.1.0-baseline.1.pt"
```

Untuk terminal baru, jalankan kembali perintah export atau tambahkan environment
variable tersebut ke konfigurasi shell lokal.

## 4. Menjalankan dashboard

```bash
streamlit run app.py
```

Browser akan membuka dashboard lokal. Pilih mode gambar atau video, unggah satu
media, isi metadata opsional, kemudian tekan **Mulai analisis**.

Metadata opsional:

- lokasi;
- waktu observasi;
- cuaca;
- catatan.

Metadata dibatasi panjangnya dan tidak diperlukan untuk inference.

## 5. Batas media

### Gambar

- format JPEG atau PNG;
- maksimal 20 MB;
- safety limit 80 megapixel;
- dinormalisasi maksimal 25 megapixel untuk pemrosesan.

### Video

- format MP4 dengan codec H.264;
- maksimal 200 MB;
- maksimal 60 detik;
- safety limit 8K dan 33.177.600 pixel/frame;
- dinormalisasi maksimal 1920×1080;
- dianalisis hingga 5 FPS.

Normalisasi mempertahankan aspect ratio. Media di luar safety limit ditolak
agar decode tidak menghabiskan resource secara tidak terkendali.

## 6. Membaca hasil

### Objek unik

Pada gambar, angka berasal dari detection valid. Pada video, angka berasal dari
confirmed track dan tidak sama dengan jumlah bbox seluruh frame.

### Komposisi jumlah

Persentase jumlah objek unik per kelas. Denominator hanya empat kelas yang
didukung.

### Proporsi area deteksi

Perbandingan area bounding box, bukan segmentasi dan bukan ukuran fisik.

### Priority

Score 0–100 berasal dari coverage, count, dan mobility share. Jika tidak ada
objek valid, status `not_assessable`.

### Recommendation

Setiap rekomendasi menunjukkan rule ID, observed value, operator, threshold,
reason, dan action. Rekomendasi tidak dibuat oleh model bahasa.

### Quality flags

- `NO_VALID_OBJECT`: tidak ada objek kelas aktif yang dinilai.
- `UNSUPPORTED_TARGET_CLASS`: taxonomy memuat kelas yang belum didukung.
- `CLASSIFICATION_UNCERTAIN`: vote kelas track kurang stabil.
- `TRACKING_UNSTABLE`: banyak track bertahan singkat.
- `LOW_ANALYSIS_QUALITY`: terdapat evidence kualitas media/decode yang rendah.

## 7. Export

JSON menyimpan hasil kanonis schema `1.0.0`, termasuk versi model, snapshot,
konfigurasi, compositions, priority, recommendations, flags, dan timings.

CSV memiliki satu baris untuk setiap target class. Empat kelas aktif memiliki
angka; `other_floating_debris` memiliki `support_status=not_supported` dan
metrik kosong.

## 8. Offline operation

Setelah dependency dan model tersedia:

1. putuskan jaringan;
2. pastikan `RIVERSORT_MODEL_PATH` benar;
3. jalankan `streamlit run app.py`;
4. gunakan media lokal.

Application runtime tidak memanggil inference API eksternal.

## 9. Troubleshooting

| Error | Tindakan |
|---|---|
| `MODEL_NOT_READY` | Jalankan model downloader dan set `RIVERSORT_MODEL_PATH`. |
| `MODEL_CHECKSUM_MISMATCH` | Hapus weight dan download ulang dari release resmi. |
| `MODEL_CLASS_MAP_MISMATCH` | Pastikan weight adalah model RiverSort yang dipublikasikan. |
| `INVALID_MEDIA_TYPE` | Gunakan JPEG/PNG atau MP4 sesuai mode. |
| `UNSUPPORTED_CODEC` | Encode video menjadi MP4 H.264. |
| `VIDEO_TOO_LONG` | Potong video menjadi maksimal 60 detik. |
| `ENCODER_UNAVAILABLE` | Instal FFmpeg dan pastikan binary dapat dijalankan. |

Stack trace dan filesystem path tidak ditampilkan pada UI. Untuk diagnosis
lokal, periksa log terminal dan gunakan media yang tidak sensitif.
