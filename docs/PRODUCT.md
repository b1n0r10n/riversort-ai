# Product overview

## Ringkasan

RiverSort AI adalah prototype software untuk mengubah gambar atau video pendek
permukaan sungai menjadi informasi awal yang lebih konsisten bagi kegiatan
pembersihan. Sistem mendeteksi empat kategori plastik, merangkum komposisi
visual, menyusun indeks prioritas, dan menghasilkan rekomendasi berbasis aturan.

Sistem tidak menggantikan survei lapangan. Hasilnya membantu pengguna menyiapkan
inspeksi, alat, pemilahan, dan dokumentasi dengan dasar yang dapat diperiksa.

## Masalah

Dokumentasi visual sungai relatif mudah dikumpulkan, tetapi foto dan video saja
belum langsung menjawab pertanyaan operasional:

1. jenis sampah plastik apa yang dominan;
2. seberapa besar kepadatan visual yang terlihat;
3. area mana yang perlu ditinjau lebih dahulu;
4. tindakan pembersihan apa yang relevan terhadap hasil pengamatan.

Penilaian manual sulit dibuat konsisten ketika volume media bertambah. Pada
video, objek yang sama juga mudah dihitung berulang pada frame yang berbeda.

## Pengguna

Pengguna awal yang dituju:

- dinas lingkungan hidup dan pengelola sungai;
- komunitas pemantau atau pembersih sungai;
- bank sampah dan TPS3R yang terlibat dalam pemilahan;
- peneliti dan organisasi lingkungan yang melakukan observasi visual.

RiverSort AI belum diklaim telah digunakan secara nasional. Arsitekturnya
dirancang agar dapat diuji pada lokasi berbeda setelah data dan validasi
lapangan tersedia.

## Tujuan produk

- Memproses satu media secara lokal tanpa layanan inferensi eksternal.
- Menyajikan hasil deteksi yang dapat diperiksa pada media beranotasi.
- Mencegah penghitungan objek yang sama berulang kali pada video.
- Memisahkan komposisi jumlah objek dari proporsi area deteksi.
- Menjelaskan penyusun indeks prioritas.
- Menghubungkan rekomendasi dengan aturan dan evidence yang eksplisit.
- Menyampaikan keterbatasan model bersama hasil analisis.

## Fitur

### Analisis gambar

1. Pengguna mengunggah JPEG atau PNG.
2. Aplikasi memeriksa ukuran, signature, MIME, dan kemampuan decode.
3. Gambar yang terlalu besar untuk pemrosesan dinormalisasi dengan rasio aspek
   tetap.
4. YOLO11n menghasilkan bounding box dan confidence.
5. Setiap deteksi valid dihitung sebagai satu objek.
6. Domain layer menghitung komposisi, coverage, priority, recommendation, dan
   quality flags.
7. Aplikasi menampilkan preview serta menyediakan JSON dan CSV.

### Analisis video

1. Pengguna mengunggah MP4 H.264 berdurasi paling lama 60 detik.
2. Video diverifikasi dengan FFprobe dan dinormalisasi maksimal 1080p.
3. Frame dianalisis secara kronologis hingga 5 FPS.
4. ByteTrack menghubungkan deteksi menjadi track.
5. Track dengan sedikitnya dua observasi dihitung satu kali.
6. Kelas akhir track ditentukan dengan confidence-weighted vote.
7. Pass kedua membuat preview H.264 dengan label akhir yang konsisten.

### Hasil

- jumlah objek unik atau track terkonfirmasi;
- komposisi berdasarkan jumlah;
- komposisi berdasarkan area deteksi;
- indeks prioritas dan tier;
- kontribusi coverage, count, dan mobility;
- rekomendasi, nilai observasi, operator, threshold, alasan, dan tindakan;
- quality flags dan keterbatasan;
- annotated media, JSON, dan CSV.

## Requirement fungsional

| ID | Requirement |
|---|---|
| F-01 | Menerima satu gambar JPEG/PNG atau video MP4 H.264. |
| F-02 | Menormalisasi resolusi media yang masih berada dalam safety limit. |
| F-03 | Mendeteksi empat kelas yang didukung capability profile. |
| F-04 | Menghitung gambar dari deteksi valid dan video dari unique track. |
| F-05 | Menampilkan dua jenis komposisi visual secara terpisah. |
| F-06 | Menghasilkan priority assessment yang dapat direkonstruksi. |
| F-07 | Menghasilkan rekomendasi deterministik dengan evidence. |
| F-08 | Menandai kelas target yang belum didukung sebagai `not_supported`. |
| F-09 | Menampilkan media beranotasi dan quality flags. |
| F-10 | Mengekspor satu hasil yang konsisten ke JSON dan CSV. |
| F-11 | Membersihkan workspace sementara setelah berhasil atau gagal. |
| F-12 | Menangani media tidak valid dengan error code dan recovery action. |

## Requirement nonfungsional

- **Traceability:** versi schema, model, snapshot, dan konfigurasi disertakan
  pada hasil.
- **Offline operation:** inference tidak membutuhkan jaringan setelah model
  tersedia.
- **Privacy:** media hanya berada dalam workspace sementara selama analisis.
- **Determinism:** formula dan rule engine tidak bergantung pada model bahasa.
- **Portability:** konfigurasi memakai path relatif atau environment variable.
- **Testability:** domain logic tidak mengimpor Streamlit, OpenCV, atau
  Ultralytics.
- **Safety:** hasil kosong tidak boleh diartikan sebagai sungai bersih.

## Batas scope

RiverSort AI tidak melakukan:

- estimasi berat, massa, tonase, atau volume fisik;
- deteksi debris nonplastik pada model aktif;
- live CCTV atau pemrosesan kamera permanen;
- kontrol alat, robot, kapal, atau pembersihan otomatis;
- keputusan lapangan tanpa verifikasi manusia;
- klaim generalisasi nasional.

Istilah yang digunakan untuk keluaran produk:

> **Estimasi komposisi visual berdasarkan jumlah objek unik dan proporsi area
> deteksi.**
