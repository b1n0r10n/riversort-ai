# Active technical decisions

Dokumen ini merangkum keputusan yang berlaku pada release `0.1.0`.

## D-01 — Software-first local prototype

RiverSort AI berjalan sebagai dashboard lokal dan tidak membutuhkan kamera
permanen, IoT, atau inference API. Keputusan ini menjaga deployment sederhana
dan memungkinkan penggunaan pada koneksi terbatas.

## D-02 — Modular monolith

Satu proses aplikasi dibagi menjadi domain, application, adapters,
infrastructure, dan presentation. Microservice, database server, dan message
queue tidak digunakan karena belum ada kebutuhan skala yang membenarkannya.

## D-03 — Capability profile empat kelas

Model aktif mendukung empat kelas plastik. `other_floating_debris` tetap berada
dalam target taxonomy tetapi berstatus `not_supported`. Nilai null membedakan
kelas yang belum dinilai dari kelas yang dinilai dan berjumlah nol.

## D-04 — FPWR sebagai satu-satunya baseline source

FPWR v6 lolos staging dan menjadi sumber snapshot baseline. River Trash tidak
digunakan untuk localization metric karena bbox export tidak merepresentasikan
objek. Sumber baru memerlukan snapshot dan model version baru.

## D-05 — Group-aware split

Duplicate dan sequence groups tidak boleh tersebar antar split. Rasio 70/15/15
merupakan target; integritas group lebih penting daripada memaksa rasio tepat.

## D-06 — Unique-track counting

Jumlah objek video berasal dari confirmed unique track, bukan total detection
per frame. Minimum dua observation mengurangi track sesaat.

## D-07 — Two-pass video rendering

Pass pertama menentukan track dan kelas akhir. Pass kedua membuat preview
dengan label akhir yang konsisten. Raw prediction di bawah threshold presentasi
boleh membantu association tetapi tidak ditampilkan sebagai hasil pasti.

## D-08 — Dua metrik komposisi

Object-count share dan detected-area share disajikan terpisah. Bounding box
tidak diperlakukan sebagai segmentasi, berat, atau volume.

## D-09 — Transparent priority and recommendations

Priority memakai formula terversi. Recommendation memakai rule engine
deterministik dan membawa evidence. Model bahasa tidak digunakan untuk
menentukan tindakan.

## D-10 — Explicit quality uncertainty

Klasifikasi dan tracking yang tidak stabil menghasilkan quality flags. Sistem
tidak menyembunyikan kondisi ketika count harus dibaca sebagai indikatif.

## D-11 — Local temporary media

Upload disimpan hanya dalam workspace analisis dan dibersihkan setelah success
atau exception. Local path tidak boleh masuk schema export atau log publik.

## D-12 — AGPL distribution

Source code dan fine-tuned model dipublikasikan di bawah
AGPL-3.0-or-later untuk memenuhi distribusi open-source dari stack Ultralytics.
Dataset tetap mengikuti CC BY 4.0 dan tidak direplikasi dalam repository.
