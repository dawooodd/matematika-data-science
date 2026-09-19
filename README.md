# 🧮 Penerapan Matematika & Probabilitas dalam Data Science

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-penjelasan__awam-orange.svg)](docs/penjelasan_matematika_awam.md)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()

Selamat datang di repositori proyek **Matematika Data Science**! 

Repositori ini mendemonstrasikan bagaimana konsep-konsep matematika tingkat lanjut, teori probabilitas, dan statistika inferensial diterjemahkan menjadi **solusi bisnis nyata berbasis data (*data-driven decision making*)**. 

Seluruh algoritma dan pengujian statistik dalam proyek ini dirancang murni dari fondasi matematika (*from scratch*) tanpa bergantung pada *black-box machine learning libraries*, memberikan transparansi penuh atas setiap angka, rumus, dan logika keputusan yang dihasilkan.

---

## 🏛️ Struktur Proyek

Repositori telah distrukturkan secara profesional dan modular agar mudah dinavigasi oleh pengembang, peneliti, maupun pemangku kepentingan bisnis:

```plaintext
matematika-untuk-data-science/
│
├── data/
│   └── .gitkeep                               # Folder untuk penyimpanan dataset sintetik
│
├── notebooks/                                 # Jupyter Notebooks interaktif untuk eksplorasi
│   ├── 01_distribusi_dan_naive_bayes.ipynb
│   ├── 02_ab_testing_sistem_rekomendasi_ttest.ipynb
│   └── 03_ab_testing_conversion_rate_ztest.ipynb
│
├── scripts/                                   # Skrip Python modular siap pakai (standalone)
│   ├── naive_bayes_from_scratch.py
│   ├── ab_testing_recommendation.py
│   └── ab_testing_conversion.py
│
├── docs/                                      # Dokumentasi konseptual mendalam
│   └── penjelasan_matematika_awam.md          # Panduan matematika & statistik ramah orang awam
│
├── .gitignore                                 # Mengabaikan cache & file sementara
├── requirements.txt                           # Daftar pustaka Python pendukung
└── README.md                                  # Dokumentasi utama proyek
```

---

## 📖 Ringkasan Tiga Studi Kasus Utama

| Studi Kasus | Domain Masalah | Metrik Evaluasi | Metodologi Matematika | Dampak Bisnis Utama |
| :--- | :--- | :--- | :--- | :--- |
| **Kasus 1: Klasifikasi Burung** | Computer Vision / Biologi Sintetik | Akurasi Prediksi Spesies (3 Ras) | Distribusi Uniform, Gaussian, Binomial, & Naive Bayes *from scratch* | Efisiensi komputasi tinggi, model *explainable* tanpa *black-box ML* |
| **Kasus 2: Sistem Rekomendasi** | Media Digital / Blog Teknologi | `articles_read` (Rata-rata artikel/sesi) | Welch's Two-Sample T-Test & Satterthwaite Degrees of Freedom | +13.98% peningkatan konsumsi konten, mendongkrak *Ad Revenue* |
| **Kasus 3: Fitur Retensi Aplikasi** | Produk Edukasi / SaaS | `retention_rate` (Proporsi pengguna aktif) | Power Analysis, Sample Size Planning, & Two-Proportion Z-Test | Kenaikan retensi +3%, menekan *churn*, efisiensi biaya uji coba 8 hari |

---

## 🔬 Bedah Studi Kasus Mendalam

### 1. Klasifikasi Ras Burung Menggunakan Teori Distribusi & Naive Bayes *From Scratch*
📂 **Notebook:** [`notebooks/01_distribusi_dan_naive_bayes.ipynb`](notebooks/01_distribusi_dan_naive_bayes.ipynb)  
🐍 **Skrip Python:** [`scripts/naive_bayes_from_scratch.py`](scripts/naive_bayes_from_scratch.py)

#### 🎯 Masalah Bisnis
Bagaimana cara sistem mengklasifikasikan spesies atau kategori objek alamiah secara akurat ketika data lapangan terbatas, tanpa harus menggunakan model *deep learning* yang boros daya komputasi dan sulit dijelaskan alasannya (*black-box*)?

#### 💡 Analogi Kehidupan Sehari-hari
> **Analogi Menyortir Buah di Pasar:**  
> Bayangkan Anda menutup mata dan diminta menebak buah di tangan Anda.
> * Anda meraba teksturnya: berkulit tebal dan berduri tajam.
> * Anda menimbang beratnya: sekitar 2 kilogram.
> * Anda mencium aromanya: menyengat manis dan khas.
> 
> Otak Anda secara otomatis menghubungkan tiap ciri fisik tersebut: *"Peluang buah berbau tajam dan berduri adalah Durian (90%), Nangka (9%), Salak (1%)"*. Algoritma **Naive Bayes** bekerja persis seperti logika otak Anda: ia menghitung peluang dari masing-masing ciri fisik secara independen, lalu memilih spesies dengan skor probabilitas akhir tertinggi.

#### 📐 Metodologi & Konsep Matematika
1. **Pembangkitan Data Sintetis (Inverse Transform Sampling):**
   Memanfaatkan fungsi inversi kumulatif (*Quantile Function / Inverse CDF*) untuk menciptakan data tiruan yang mengikuti pola alamiah:
   * **Distribusi Gaussian / Normal ($\mu, \sigma$):** Dimodelkan untuk fitur kontinu fisik seperti lebar sayap (`wingspan_cm`) dan berat badan (`weight_g`).
     $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$
   * **Distribusi Binomial ($n, p$):** Dimodelkan untuk frekuensi kejadian diskrit berulang, yaitu jumlah hari burung berkicau (`sing_days`) dalam 30 hari pemantauan.
     $$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$
   * **Distribusi Uniform ($a, b$):** Dimodelkan untuk rasio paruh terhadap kepala (`beak_head_ratio`) yang memiliki peluang merata pada rentang tertentu.
     $$f(x) = \frac{1}{b - a}$$
2. **Klasifikasi Naive Bayes:**
   Menghitung peluang *Posterior* untuk setiap kelas ras burung $C_k$ berdasarkan data fitur $X = (x_1, x_2, \dots, x_d)$:
   $$P(C_k \mid X) \propto P(C_k) \prod_{j=1}^{d} P(x_j \mid C_k)$$
   Di mana $P(C_k)$ adalah probabilitas awal (*Prior*), dan $\prod P(x_j \mid C_k)$ adalah peluang kemunculan fitur (*Likelihood*) yang dihitung menggunakan fungsi kepekatan peluang (PDF/PMF) dari distribusi masing-masing fitur. Prediksi akhir ditentukan oleh nilai posterior terbesar (*Maximum A Posteriori / MAP*).

#### 💼 Dampak Bisnis (Business Impact)
* **Explainability (Kepatuhan & Transparansi):** Model tidak bekerja seperti *black-box*. Manajemen dan klien dapat melihat secara transparan fitur apa yang paling berkontribusi terhadap klasifikasi.
* **Efisiensi Biaya Infrastruktur:** Algoritma Naive Bayes membutuhkan memori yang sangat kecil dan dapat dijalankan pada perangkat komputasi hemat daya (*edge computing* / IoT di lapangan) tanpa GPU mahal.
* **Generalisasi ke Industri:** Pendekatan ini dapat langsung diadaptasi untuk deteksi *spam/phishing* email di perbankan, diagnosis medis awal, dan klasifikasi keluhan tiket pelanggan otomatis.

---

### 2. Evaluasi Fitur "Sistem Rekomendasi Artikel" (A/B Testing - Welch's T-Test)
📂 **Notebook:** [`notebooks/02_ab_testing_sistem_rekomendasi_ttest.ipynb`](notebooks/02_ab_testing_sistem_rekomendasi_ttest.ipynb)  
🐍 **Skrip Python:** [`scripts/ab_testing_recommendation.py`](scripts/ab_testing_recommendation.py)

#### 🎯 Masalah Bisnis
Sebuah media blog teknologi meluncurkan fitur *widget* "Artikel Terkait" di bagian bawah halaman. Manajemen ingin memastikan: **apakah penambahan fitur ini secara nyata mendorong pembaca untuk mengonsumsi lebih banyak konten, ataukah peningkatan yang terlihat hanya kebetulan musiman belaka?**

#### 💡 Analogi Kehidupan Sehari-hari
> **Analogi Menguji Resep Baru di Restoran:**  
> Anda adalah pemilik restoran yang menguji apakah sambal baru membuat pengunjung makan lebih banyak nasi. 
> * 50 pengunjung diberi sambal lama (Kelompok Origin).
> * 50 pengunjung diberi sambal baru (Kelompok Varian).
> 
> Anda menghitung rata-rata piring nasi yang dihabiskan. Uji **T-Test** berfungsi seperti juri independen: ia mengecek apakah perbedaan piring nasi yang habis memang karena sambal baru lebih lezat, atau hanya karena kebetulan pelanggan hari itu sedang lapar. Karena nafsu makan tiap orang sangat bervariasi (*varians tidak sama*), kita menggunakan **Welch's T-Test** yang tidak menuntut variasi nafsu makan kedua kelompok harus sama rata.

#### 📐 Metodologi & Konsep Matematika
* **Metrik Pengujian:** `articles_read` (jumlah artikel yang dibaca per sesi, berskala numerik kontinu).
* **Desain Eksperimen:** Dua sampel independen selama 20 hari (Kelompok Origin vs Kelompok Varian).
* **Welch-Satterthwaite Degrees of Freedom ($df$):**
  Mengoreksi derajat kebebasan untuk mengantisipasi ketidaksamaan varians ($\sigma_1^2 \ne \sigma_2^2$):
  $$df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$
* **Welch's T-Statistic:**
  $$t = \frac{\bar{x}_{\text{varian}} - \bar{x}_{\text{origin}}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
* **Uji Hipotesis Dua Sisi (Two-Tailed P-Value):**
  $$p\text{-value} = 2 \times \left(1 - F_t(|t|, df)\right)$$
  Jika $p\text{-value} < \alpha$ ($\alpha = 0.05$), maka Hipotesis Nol ($H_0$) ditolak.

#### 💼 Dampak Bisnis (Business Impact)
* **Peningkatan Engagement Signifikan:** Kelompok Varian membaca rata-rata **5.38 artikel/sesi**, meningkat dibandingkan Kelompok Origin yang membaca **4.72 artikel/sesi** (peningkatan absolut $+0.66$ artikel, atau **$+13.98\%$**).
* **P-Value Signifikan ($p < 0.05$):** Membuktikan secara ilmiah bahwa fitur rekomendasi adalah penyebab utama kenaikan keterlibatan pembaca, bukan faktor kebetulan.
* **Monetisasi Digital & Ad Revenue:** Peningkatan konsumsi artikel sebesar ~14% secara linier mendongkrak *ad impressions*, memperpanjang waktu singgah (*dwell time*), dan menurunkan *bounce rate*, sehingga melipatgandakan pendapatan iklan tanpa menambah biaya akuisisi pengunjung baru (*Zero CAC scaling*).
* **Rekomendasi Tindakan:** Rilis fitur "Artikel Terkait" ke 100% pengguna web.

---

### 3. Peningkatan Retensi Aplikasi Edukasi (A/B Testing - Two-Proportion Z-Test)
📂 **Notebook:** [`notebooks/03_ab_testing_conversion_rate_ztest.ipynb`](notebooks/03_ab_testing_conversion_rate_ztest.ipynb)  
🐍 **Skrip Python:** [`scripts/ab_testing_conversion.py`](scripts/ab_testing_conversion.py)

#### 🎯 Masalah Bisnis
Perusahaan edutech merombak desain antarmuka (UI) aplikasi belajar mereka dengan harapan meningkatkan retensi pengguna dari baseline **69%** menjadi minimal **72%** (efek minimum terdeteksi / MDE = $+3\%$). Manajemen membutuhkan jawaban atas dua pertanyaan krusial:
1. *Berapa jumlah pengguna dan berapa hari pengujian yang dibutuhkan agar hasil tes valid dan tidak bias?*
2. *Apakah UI baru tersebut terbukti sukses secara statistik?*

#### 💡 Analogi Kehidupan Sehari-hari
> **Analogi Survei Minuman Kaleng Berskala Nasional:**  
> Jika Anda hanya bertanya kepada 10 orang apakah mereka suka rasa minuman baru, hasilnya bisa sangat bias (kebetulan dapat 8 orang yang suka manis). 
> 
> Sebelum survei dimulai, Anda menghitung dulu berapa ratus orang yang harus ditanya agar hasilnya mencerminkan seluruh masyarakat (*Sample Size Calculation*). Karena jawabannya hanya biner: **Suka (1)** atau **Tidak Suka (0)** dan jumlah orangnya ribuan, Anda menggunakan kurva lonceng normal standar (**Z-Test**) untuk memastikan persentase keunggulan rasa baru tersebut dapat dipercaya.

#### 📐 Metodologi & Konsep Matematika
1. **Analisis Kekuatan Statistik & Estimasi Ukuran Sampel (*Power Analysis*):**
   * Baseline retensi: $p_1 = 0.69$, Target varian: $p_2 = 0.72$.
   * $\alpha = 0.05$ (Tingkat signifikansi 5%, Confidence Level 95%).
   * $\beta = 0.20$ (Power Statistik $1 - \beta = 80\%$, memastikan kemampuan mendeteksi dampak nyata).
   * Rumus ukuran sampel per kelompok:
     $$n = \frac{\left( Z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + Z_{\beta}\sqrt{p_1(1-p_1) + p_2(1-p_2)} \right)^2}{(p_2 - p_1)^2}$$
     Diperoleh kebutuhan sampel: **$3.627$ pengguna per kelompok** (total $7.254$ pengguna).
2. **Kalkulasi Durasi Eksperimen:**
   Dengan jumlah pengguna aktif harian (*Daily Active Users / DAU*) sebanyak **$997$ pengguna/hari**:
   $$\text{Durasi} = \left\lceil \frac{7.254}{997} \right\rceil = 8 \text{ hari}$$
3. **Uji Hipotesis Dua Proporsi (Pooled Z-Test):**
   * Proporsi gabungan (*Pooled Proportion*):
     $$\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}$$
   * Z-Statistic untuk selisih proporsi:
     $$Z = \frac{\hat{p}_2 - \hat{p}_1}{\sqrt{\hat{p}(1 - \hat{p}) \left( \frac{1}{n_1} + \frac{1}{n_2} \right)}}$$
   * Evaluasi $p\text{-value}$ dari fungsi distribusi normal kumulatif standar ($\Phi$):
     $$p\text{-value} = 2 \times (1 - \Phi(|Z|))$$

#### 💼 Dampak Bisnis (Business Impact)
* **Pencegahan Eksperimen Sia-sia (*Cost Optimization*):** Mengetahui durasi tepat 8 hari mencegah tim menghentikan tes terlalu cepat (*underpowered test*) atau menjalankan tes terlalu lama yang memboroskan biaya *server* dan memperlambat *time-to-market*.
* **Peningkatan Retensi Nyata:** UI baru mencatatkan tingkat retensi sebesar **71.9%** (berbanding 69% pada kontrol) dengan hasil uji hipotesis menolak $H_0$ ($p < 0.05$).
* **Nilai Finansial Seumur Hidup Pelanggan (*Customer Lifetime Value / LTV*):** Kenaikan retensi sebesar 3% pada platform edukasi dengan ratusan ribu pengguna secara langsung menurunkan angka *user churn*, mempertahankan langganan aktif berulang, dan meningkatkan ROI pemasaran secara berkesinambungan.

---

## 🛠️ Tumpukan Teknologi (Tech Stack)

* **Bahasa Pemrograman:** Python 3.9+
* **Komputasi Numerik & Matriks:** `NumPy`
* **Manajemen & Analisis Tabular:** `Pandas`
* **Probabilitas & Statistik Terapan:** `SciPy` (`scipy.stats`, `scipy.special`)
* **Visualisasi & Eksplorasi Data:** `Matplotlib`, `Seaborn`
* **Lingkungan Interaktif:** `Jupyter Notebook`

---

## 🚀 Panduan Instalasi & Eksekusi

### 1. Kloning Repositori
```bash
git clone https://github.com/username/matematika-untuk-data-science.git
cd matematika-untuk-data-science
```

### 2. Buat & Aktifkan Virtual Environment (Direkomendasikan)
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalasi Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Menjalankan Kode
Anda dapat menjalankan studi kasus melalui dua cara:

* **Opsi A: Melalui Jupyter Notebook (Interaktif & Visual)**
  ```bash
  jupyter notebook
  ```
  Buka folder `notebooks/` di browser Anda dan jalankan berkas notebook berurutan (01, 02, atau 03).

* **Opsi B: Melalui Terminal (Skrip Python Standalone)**
  ```bash
  # Menjalankan simulasi Naive Bayes from scratch
  python scripts/naive_bayes_from_scratch.py

  # Menjalankan simulasi A/B testing t-test
  python scripts/ab_testing_recommendation.py

  # Menjalankan simulasi A/B testing z-test
  python scripts/ab_testing_conversion.py
  ```

---

## 📚 Pelajari Lebih Lanjut: Panduan Bahasa Awam

Ingin memahami lebih dalam bagaimana cara membedakan **kapan harus memakai T-Test atau Z-Test**, cerita penemuan uji statistik di pabrik bir, serta analogi konsep PDF, CDF, dan *P-Value* tanpa pusing dengan rumus?

👉 **Baca panduan lengkapnya di:** [**`docs/penjelasan_matematika_awam.md`**](docs/penjelasan_matematika_awam.md)

---

## 👨‍💻 Kontributor & Portofolio

Proyek ini disusun sebagai bagian dari portofolio profesional penerapan matematika dan statistika inferensial dalam Data Science modern.

* **Penulis / Data Scientist:** Muchammad Nasich
* **Lisensi:** Didistribusikan di bawah lisensi MIT. Silakan gunakan dan kembangkan untuk tujuan edukasi dan portofolio profesional.
