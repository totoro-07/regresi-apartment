# 🏢 Prediksi Harga Sewa Apartemen

Proyek ini bertujuan untuk membangun model prediktif berbasis machine learning guna memperkirakan harga sewa apartemen berdasarkan fitur-fitur properti seperti luas bangunan, jumlah kamar, kamar mandi, dan lokasi.

---

## 📌 1. Domain Proyek

Harga sewa properti merupakan variabel yang sangat dipengaruhi oleh banyak faktor: lokasi, ukuran, jumlah kamar, dan fasilitas. Penetapan harga secara manual seringkali tidak objektif dan rentan terhadap kesalahan estimasi.

Dalam proyek ini, kami mengangkat isu tersebut dan membangun sistem prediktif menggunakan data historis sebagai dasar penetapan harga.

> Referensi:  
> Si, R. & Lu, Min & Arikawa, Masatoshi & Asami, Yasushi & Iwasaki, J.. (2014). Finding Good Areas for Renting Apartments Using Apartments Information and Users' Trajectories. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. XL-4. 10.5194/isprsarchives-XL-4-229-2014.

> Sirmans, G. & Benjamin, John. (1991). Determinants of Market Rent. Journal of Real Estate Research. 6. 357-380.
10.1080/10835547.1991.12090653.
---

## 🎯 2. Business Understanding

1. Harga sewa yang ditampilkan pada iklan sering kali tidak mencerminkan nilai pasar yang sebenarnya.
2. Pengguna (pemilik atau penyewa) kesulitan untuk menentukan apakah harga sewa suatu apartemen tergolong mahal atau murah berdasarkan karakteristiknya.

### Goals
1. Membangun model machine learning untuk memprediksi harga sewa apartemen berdasarkan fitur properti.

2. Membandingkan performa beberapa model regresi untuk menemukan model terbaik.

3. Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap harga sewa apartemen.

### Solution Statement
- **Ridge Regression** dan **Lasso Regression**  
  Digunakan sebagai model baseline. Memberikan interpretasi yang sederhana dan cepat dilatih, serta dapat menangani multikolinearitas.
- **Random Forest Regressor**, **Gradient Boosting**, **XGBoost**, **CatBoost**, dan **LightGBM**  
  Termasuk Random Forest, Gradient Boosting, XGBoost, CatBoost, dan LightGBM untuk menangkap hubungan non-linear antar fitur dan target.
- **RMSE (Root Mean Squared Error)** – RMSE digunakan sebagai metrik utama untuk mengukur akurasi prediksi.

## 📊 3. Data Understanding

## 3.Data Understanding

Dataset diambil dari [Apartments for Rent Classified](https://www.kaggle.com/datasets/adithyaawati/apartments-for-rent-classified).

### Informasi Dataset
- **File:** `apartments_for_rent_classified_10K.csv`
- **Jumlah data:** 10.000 baris
- **Jumlah Kolom:**  22

### Deskripsi Fitur:
| Fitur                          | Deskripsi                                     |
| ------------------------------ | --------------------------------------------- |
| `id`                           | ID unik untuk setiap listing apartemen        |
| `amenities`                    | Daftar fasilitas yang disediakan              |
| `currency`                     | Mata uang yang digunakan untuk harga          |
| `fee`                          | Biaya tambahan selain harga sewa utama        |
| `has_photo`                    | Boolean, apakah terdapat foto di listing      |
| `pets_allowed`                 | Boolean, apakah hewan peliharaan diizinkan    |
| `price_display`                | Format tampilan harga di listing              |
| `price_type`                   | Tipe harga, misalnya bulanan atau tahunan     |
| `address`                      | Alamat apartemen                              |
| `cityname`                     | Nama kota tempat apartemen berada             |
| `state`                        | Negara bagian atau wilayah                    |
| `latitude`, `longitude`        | Koordinat lokasi geografis apartemen          |
| `source`                       | Sumber atau platform pengiklan                |
| `time`                         | Timestamp kapan data diambil                  |
| `price`                        | Harga sewa apartemen (target)                 |
| `bedrooms`                     | Jumlah kamar tidur                            |
| `bathrooms`                    | Jumlah kamar mandi                            |
| `sqfeet`                       | Luas apartemen dalam satuan kaki persegi      |
| `cats_allowed`, `dogs_allowed` | Apakah kucing atau anjing diizinkan           |
| `region`                       | Wilayah administratif tempat apartemen berada |
| `type`                         | Jenis properti, misal apartment, condo, dsb   |


### Kondisi Data
- Terdapat missing value pada beberapa kolom seperti amenities dan fee
- Distribusi harga sewa sangat skewed dan mengandung outlier


### EDA *(Rubrik Tambahan)*
- Visualisasi distribusi price dan sqfeet

- Korelasi antara bedrooms, bathrooms, dan price

- Analisis frekuensi kata dalam kolom amenities


---

## 🧹 4. Data Preparation

1. Penanganan Missing Value:
Baris dengan nilai kosong penting dihapus menggunakan dropna().

Fitur teks seperti amenities, type, dan fee diisi dengan nilai "tidak tersedia" menggunakan fillna("tidak tersedia").

2. Penghapusan Kolom Tidak Relevan:
Beberapa fitur dihapus karena tidak relevan atau redundan, seperti: id, time, address, price_display, price_type, currency, source.

3. Encoding Fitur Kategorikal:
Fitur cityname, state, type, dan region dikodekan menggunakan One-Hot Encoding (pd.get_dummies()), bukan Label Encoding.

4. Transformasi Variabel Target:
Variabel target price ditransformasikan menggunakan PowerTransformer untuk mengurangi skewness distribusi.

5. Splitting Data:
Dataset dibagi menjadi data latih dan data uji menggunakan train_test_split dengan rasio 80:20.

6. Normalisasi Data:
Data numerik seperti bedrooms, bathrooms, sqfeet, dst., dinormalisasi menggunakan StandardScaler untuk meningkatkan performa model.


---

## 🤖 5. Modeling

## 🧠 Algoritma yang Digunakan

### 🔹 K-Nearest Neighbors (KNN)
- Memprediksi harga berdasarkan rata-rata harga dari **K tetangga terdekat** dalam ruang fitur.
- **Parameter yang digunakan:**  
  `n_neighbors = 4`

---

### 🔹 Random Forest Regressor
- Model **ensemble** yang terdiri dari banyak pohon keputusan.
- Hasil prediksi diambil dari rata-rata prediksi seluruh pohon.
- **Parameter:**  
  Menggunakan parameter default, dengan tuning awal pada:  
  `n_estimators`, `max_depth`

---

### 🔹 XGBoost Regressor
- Model **gradient boosting** yang membangun pohon secara bertahap untuk meminimalkan kesalahan dari model sebelumnya.
- **Parameter:**  
  `verbosity = 0`, `n_estimators`, `learning_rate`, `max_depth`  
  (tuning disiapkan namun belum sepenuhnya diselesaikan)

---

### 🔹 CatBoost Regressor
- Model **boosting** yang efisien dalam menangani fitur kategorikal tanpa perlu pra-pemrosesan tambahan.
- **Parameter:**  
  `verbose = 0`

---

### 🛠️ Tuning Parameter
- Proses **GridSearchCV** telah dicoba untuk beberapa model, namun tidak seluruhnya ditampilkan dalam notebook akhir.
- **Rekomendasi ke depan:** Melakukan tuning parameter lebih lanjut secara sistematis untuk mengoptimalkan performa model.


## 📏 Evaluation

### 🧪 Metrik Evaluasi yang Digunakan

Untuk mengukur performa model regresi, digunakan tiga metrik utama:

- **RMSE (Root Mean Squared Error)**: Mengukur seberapa besar rata-rata error prediksi. Semakin kecil nilainya, semakin baik.
- **MAE (Mean Absolute Error)**: Rata-rata selisih absolut antara nilai aktual dan prediksi.
- **R² Score (Koefisien Determinasi)**: Mengukur proporsi variansi target yang dapat dijelaskan oleh model. Nilai mendekati 1 berarti model sangat baik.

---

### 📋 Hasil Evaluasi Model

| No | Model              | RMSE     | MAE      | R² Score  |
|----|--------------------|----------|----------|-----------|
| 1  | XGBoost            | 0.5191   | 0.3762   | 0.7245    |
| 2  | CatBoost           | 0.5241   | 0.3838   | 0.7192    |
| 3  | LightGBM           | 0.5501   | 0.3991   | 0.6907    |
| 4  | Gradient Boosting  | 0.6161   | 0.4645   | 0.6120    |
| 5  | Lasso              | 0.9890   | 0.7774   | -0.0000   |
| 6  | Random Forest      | 0.5351   | 0.3802   | 0.7073    |
| 7  | Bayesian Ridge     | 0.5766   | 0.4080   | 0.6601    |
| 8  | Support Vector     | 0.6373   | 0.4530   | 0.5849    |
| 9  | K-Nearest Neighbors| 0.7499   | 0.5507   | 0.4252    |
|10  | Ridge              | 0.6077   | 0.4203   | 0.6225    |

---

### 🏆 Kesimpulan Evaluasi

- **Model terbaik berdasarkan RMSE dan R² Score adalah XGBoost**, yang menunjukkan performa terbaik dalam memprediksi harga sewa apartemen.
- **CatBoost dan Random Forest** juga memberikan performa yang cukup kompetitif.
- **Lasso Regression** memiliki performa terburuk dengan nilai R² negatif, menandakan bahwa model tersebut bahkan lebih buruk dari rata-rata konstan.
- Pemilihan model akhir didasarkan pada kombinasi RMSE paling rendah dan R² paling tinggi.
![Actual vs Predicted Values (XGBoost)](![alt text](image-1.png))

---


