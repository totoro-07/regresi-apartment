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

1. Harga sewa apartemen yang tercantum di iklan seringkali tidak mencerminkan nilai pasar aktual.
2. Sulit bagi pemilik atau penyewa untuk menentukan apakah harga tersebut wajar atau tidak.

### Goals
Membangun model machine learning untuk memprediksi harga sewa apartemen berdasarkan fitur-fitur seperti lokasi, ukuran, jumlah kamar, dan deskripsi properti.


### Solution Statement
- **Ridge Regression** dan **Lasso Regression**  
  Model regresi linier dengan regularisasi (L2 dan L1) digunakan sebagai baseline. Model ini mudah diinterpretasikan dan cepat dalam pelatihan, serta mampu menangani multikolinearitas.
- **Random Forest Regressor**, **Gradient Boosting**, **XGBoost**, **CatBoost**, dan **LightGBM**  
  Kelompok model ensemble yang mampu menangkap hubungan non-linear antara fitur dan harga sewa. XGBoost dan LightGBM dikenal cepat dan akurat, sementara CatBoost menangani fitur kategori secara efisien.
- **RMSE (Root Mean Squared Error)** – untuk mengukur kesalahan prediksi

## 📊 3. Data Understanding

## 3.Data Understanding

Dataset diambil dari [Apartments for Rent Classified](https://www.kaggle.com/datasets/adithyaawati/apartments-for-rent-classified).

**Informasi Dataset**

- File: apartments_for_rent_classified_100K.csv

### Informasi Dataset
- **File:** `apartments_for_rent_classified_100K.csv`
- **Jumlah data:** ~100.000 baris
- **Fitur utama:** `price`, `bedrooms`, `bathrooms`, `size_sqft`, `location`, `description`, `category`

### Kondisi Data
- Terdapat missing value pada beberapa kolom
- Distribusi harga tidak normal (ada outlier ekstrem)

### Penjelasan Fitur
- `price`: target prediksi
- `bedrooms`, `bathrooms`, `size_sqft`: fitur numerik
- `location`, `description`: fitur kategori/teks
- `category`: jenis unit (apartment, studio, dll)

### EDA *(Rubrik Tambahan)*
- Korelasi fitur terhadap harga
- Distribusi harga berdasarkan lokasi
- Word frequency dari deskripsi properti


---

## 🧹 4. Data Preparation

### ✅ Langkah yang dilakukan:
1. **Missing Value Handling**  
   Menghapus baris dengan nilai kosong yang signifikan.
2. **Encoding Kategorikal**  
   Fitur seperti `city` diencode dengan Label Encoding.
3. **Normalisasi**  
   Fitur numerik dinormalisasi untuk mempercepat konvergensi model.
4. **Train-Test Split**  
   Data dibagi 80:20 untuk melatih dan mengevaluasi model.


---

## 🤖 5. Modeling

Model yang dibangun:
- XGBoost
- CatBoost
- Random Forest
- LightGBM
- Bayesian Ridge
- Ridge Regression
- Gradient Boosting
- Support Vector Regression
- K-Nearest Neighbors (KNN)
- Lasso Regression
![Perbandingan RMSE Antar Model](![alt text](image.png))


### 🔧 Pemodelan & Tuning
- Model RF & GBoost dituning dengan GridSearchCV
- Parameter: `n_estimators`, `max_depth`, `learning_rate`, dll.

### 📌 Pemilihan Model Terbaik
Model Gradient Boosting dipilih karena menghasilkan error terkecil (RMSE) dan stabil terhadap data uji.

> Kelebihan/kekurangan tiap model juga dibahas pada bagian evaluasi.

---

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


