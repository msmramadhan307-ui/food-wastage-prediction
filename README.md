# 🍲 Food Wastage Prediction using Machine Learning

Proyek ini bertujuan untuk memprediksi jumlah makanan yang terbuang (`Wastage Food Amount`) dalam skala kilogram (kg) pada berbagai jenis acara. Pemodelan dibangun menggunakan pendekatan metodologi **CRISP-DM** dengan membandingkan tiga algoritma pembelajaran mesin untuk menemukan estimasi terbaik.

---

## 📊 Hasil Perbandingan Model

Evaluasi dilakukan pada data uji (*test set*) menggunakan metrik **Root Mean Squared Error (RMSE)** dan **Koefisien Determinasi ($R^2$ Score)**:

| Model | RMSE (kg) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: |
| Linear Regression | 5.39 | 0.73 | Baseline[cite: 5] |
| Gradient Boosting Regressor | 3.38 | 0.89 | Candidate[cite: 5] |
| **Random Forest Regressor** | **3.17** | **0.91** | **Best Model**[cite: 5] |

### 📈 Evaluasi Model Final (Random Forest)
Model **Random Forest Regressor** dipilih sebagai model terbaik dengan hasil evaluasi pada data uji sebagai berikut[cite: 5]:
* **$R^2$ Score:** `0.91` (mampu menjelaskan $91\%$ variansi data target)[cite: 5]
* **MAE:** `1.97` kg (rata-rata kesalahan prediksi mutlak)[cite: 5]
* **RMSE:** `3.17` kg (standar deviasi error prediksi)[cite: 5]

---

## 🔑 Temuan Utama (Feature Importance & Korelasi)

* **Korelasi Positif:** Variabel `Number of Guests` ($r = 0.64$) dan `Quantity of Food` ($r = 0.62$) memiliki korelasi kuat terhadap jumlah makanan terbuang[cite: 3].
* **Kontribusi Prediksi:** Analisis *Feature Importance* mengonfirmasi bahwa skala kapasitas acara (jumlah tamu dan pasokan makanan) menjadi fitur yang paling berpengaruh dalam keputusan prediksi model[cite: 3, 5].

---

## 📁 Struktur Repositori

```text
├── food_wastage_data.xlsx         # Dataset utama
├── train food wastage.py          # Script untuk preprocessing, eksplorasi, melatih model, & menyimpan artefak .pkl
├── predict food wastage.py        # Script khusus inferensi/prediksi data baru menggunakan file .pkl
├── food_wastage_model.pkl         # Artefak model Random Forest terpilih
├── food_wastage_preprocessor.pkl  # Artefak ColumnTransformer (Scaler & Encoder)
├── requirements.txt               # Daftar pustaka/dependency Python
└── README.md                      # Dokumentasi proyek
