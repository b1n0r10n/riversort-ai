# Roadmap

Roadmap berikut merupakan arah pengembangan, bukan klaim kemampuan pada release
sekarang. Setiap tahap memerlukan data, pengujian, dan validation gate baru.

## 1. Pilot dan data Indonesia

- Mengumpulkan media sungai dari beberapa kondisi pencahayaan, musim, dan
  kepadatan.
- Menetapkan izin, provenance, dan annotation guideline.
- Menambah negative samples seperti air, vegetasi, refleksi, dan debris alami.
- Mengukur perbedaan performa antar lokasi tanpa menggabungkan data secara
  diam-diam.

Gate: dataset terversi, lisensi jelas, group-aware split, dan audit manual.

## 2. Perbaikan detector dan dense scene

- Menambah contoh objek kecil, occluded, dan tumpukan padat.
- Mengevaluasi resolusi, augmentasi, dan ukuran model secara terkontrol.
- Membandingkan detector berdasarkan recall per kelas, false positives, latency,
  dan resource.
- Mengembangkan quality estimation yang lebih kuat untuk count indikatif.

Gate: peningkatan pada held-out data tanpa menurunkan kelas lain secara tidak
terukur.

## 3. Kelas debris nonplastik

- Menyusun definisi kelas `other_floating_debris`.
- Melakukan anotasi ulang data yang memiliki localization valid.
- Menilai confusion dengan plastik dan vegetasi.
- Membuat capability profile dan model version baru.
- Mengaktifkan rule debris hanya setelah model dipromosikan.

Gate: kelas tersedia pada train/validation/test dan lulus evaluation review.

## 4. Validasi lapangan

- Membandingkan hasil visual dengan catatan pengamatan manusia.
- Menguji apakah rekomendasi membantu pemilihan alat dan pemilahan.
- Mengukur ID fragmentation pada video lapangan.
- Mendokumentasikan failure cases dan keputusan operasional yang tidak dapat
  diturunkan dari gambar.

Gate: protocol evaluasi dan persetujuan domain reviewer.

## 5. Product readiness

- Riwayat analisis multi-lokasi.
- Paket deployment yang mudah diperbarui.
- Role dan access control jika data disimpan.
- API/backend terpisah hanya jika kebutuhan integrasi terbukti.
- Observability tanpa menyimpan media atau metadata sensitif.

Hosted deployment, kamera permanen, drone, IoT, robot, dan kapal otomatis
merupakan kemungkinan integrasi jangka panjang. Fitur tersebut bukan kebutuhan
utama RiverSort AI dan tidak mengubah posisi produk sebagai software
decision-support.
