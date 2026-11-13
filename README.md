# PlantVision — Dashboard Klasifikasi Kondisi Daun

Dashboard berbasis **Streamlit** untuk mengklasifikasikan kondisi daun ke dalam tiga kelas menggunakan model **MobileNetV2** yang dilatih melalui notebook `Daun_Klasifikasi_Colab.ipynb`.

## Kelas Klasifikasi

- `Sehat`
- `Menguning`
- `Rusak`

## Fitur

- Mengunggah gambar daun atau mengambil gambar menggunakan kamera.
- Memeriksa apakah gambar menyerupai daun.
- Mengklasifikasikan kondisi daun ke dalam tiga kelas.
- Menampilkan hasil klasifikasi dan skor confidence.
- Menyimpan riwayat hasil klasifikasi selama sesi berlangsung.
- Mengunduh riwayat klasifikasi dalam format CSV.

## Struktur Folder

```text
PlantVision/
├── app.py
├── requirements.txt
├── README.md
├── plantvision_leaf_classifier.keras
└── class_names.txt
```

## 1. Menyiapkan File Model

Jalankan notebook `Daun_Klasifikasi_Colab.ipynb` hingga bagian **Simpan model dan daftar kelas**. Unduh file berikut:

- `plantvision_leaf_classifier.keras`
- `class_names.txt`

Letakkan kedua file tersebut di folder yang sama dengan `app.py`.

**Catatan:** File model tidak dibuat otomatis oleh dashboard. Model harus dihasilkan dari proses training pada notebook terlebih dahulu.

## 2. Menjalankan di Laptop

Pastikan Python sudah terpasang, lalu buka terminal pada folder proyek dan instal dependensi:

```powershell
pip install -r requirements.txt
```

Jalankan aplikasi dengan perintah:

```powershell
streamlit run app.py
```

Aplikasi akan terbuka di browser melalui alamat lokal yang ditampilkan di terminal.

### Mengakses dari HP pada Wi-Fi yang Sama

Jalankan aplikasi menggunakan perintah:

```powershell
streamlit run app.py --server.address 0.0.0.0
```

Buka `Network URL` yang ditampilkan di terminal melalui browser HP. Jika diperlukan, izinkan Python di Windows Firewall untuk jaringan privat.

## 3. Deploy ke Streamlit Community Cloud

Aplikasi dapat dipublikasikan melalui GitHub dan Streamlit Community Cloud.

1. Unggah file proyek ke repository GitHub.
2. Pastikan `app.py` dan `requirements.txt` tersedia di repository.
3. Pastikan file model `plantvision_leaf_classifier.keras` dan `class_names.txt` dapat diakses oleh aplikasi saat dijalankan.
4. Buka [Streamlit Community Cloud](https://share.streamlit.io/), lalu masuk menggunakan akun GitHub.
5. Pilih **Create app**, kemudian tentukan repository, branch, dan file utama `app.py`.
6. Klik **Deploy** dan tunggu hingga proses instalasi dependensi serta inisialisasi aplikasi selesai.

Setelah berhasil, aplikasi akan memiliki URL publik dengan domain `streamlit.app`.

**Perhatian:** Jika ukuran file model terlalu besar untuk diunggah langsung ke GitHub, gunakan Git LFS atau penyimpanan model eksternal. Pastikan juga kebutuhan memori dan dependensi TensorFlow serta PyTorch sesuai dengan resource yang tersedia di Streamlit Community Cloud.

## Catatan Teknis

- Model menerima gambar RGB berukuran `224 × 224` piksel.
- Notebook memasukkan proses rescaling ke dalam model, sehingga dashboard tidak menjalankan `mobilenet_v2.preprocess_input` lagi.
- Nama kelas dibaca dari `class_names.txt`, dengan urutan fallback `Sehat`, `Menguning`, dan `Rusak`.
- Pemeriksaan gambar menggunakan CLIP, sedangkan klasifikasi kondisi daun menggunakan MobileNetV2.
- Deteksi kandidat daun menggunakan heuristik warna dan kontur OpenCV. Hasilnya dapat dipengaruhi oleh pencahayaan, latar belakang, dan daun yang saling bertumpuk.
- Confidence merupakan skor prediksi model, bukan diagnosis pasti mengenai kesehatan tanaman.
