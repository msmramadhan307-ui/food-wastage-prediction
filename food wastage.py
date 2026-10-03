import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Membaca dataset
df = pd.read_excel('food_wastage_data.xlsx')
df.head()

print(df.dtypes)

#Hapus Duplikat
df = df.drop_duplicates()

# Cek missing values
print("Missing values per column:")
display(df.isnull().sum())

# Daftar semua kolom numerik
numerical_features = ['Number of Guests', 'Quantity of Food', 'Wastage Food Amount']

print("Matriks Korelasi untuk semua fitur numerik:")
display(df[numerical_features].corr())

# Membuat peta kalor (heatmap) korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Number of Guests', 'Quantity of Food', 'Wastage Food Amount']].corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur Numerik')
plt.show()

# Menentukan fitur dan target
X = df.drop(columns=['Wastage Food Amount'])
y = df['Wastage Food Amount']

# Mengidentifikasi kolom kategorikal dan numerik
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

#Split Data Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Transformasi Fitur (Scaling & Encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Fit & Transform hanya pada data latih (X_train)
X_train_processed = preprocessor.fit_transform(X_train)

# Transform saja pada data uji (X_test)
X_test_processed = preprocessor.transform(X_test)

"""**MEMBANDINGKAN 3 ALGORITMA & VISUALISASI MODEL**"""

# 1. Inisialisasi 3 Algoritma
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# 2. Latih dan Evaluasi Masing-masing Model
results = []

for name, model in models.items():
    # Latih Model
    model.fit(X_train_processed, y_train)

    # Prediksi Data Uji
    y_pred = model.predict(X_test_processed)

    # Hitung Metrik Evaluasi
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Simpan Hasil
    results.append({
        'Model': name,
        'RMSE': round(rmse, 2),
        'R2 Score': round(r2, 2)
    })

# 3. Tampilkan Ringkasan Perbandingan
df_results = pd.DataFrame(results)
print(df_results)

# Visualisasi sederhana menggunakan Matplotlib
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].bar(df_results['Model'], df_results['RMSE'], color=['#a6cee3', '#1f78b4', '#b2df8a'])
ax[0].set_title('Perbandingan RMSE')
ax[0].set_ylabel('RMSE (Makin rendah makin baik)')

ax[1].bar(df_results['Model'], df_results['R2 Score'], color=['#a6cee3', '#1f78b4', '#b2df8a'])
ax[1].set_title('Perbandingan R2 Score')
ax[1].set_ylabel('R2 (Makin tinggi makin baik)')

plt.tight_layout()
plt.show()

"""**SELEKSI & EVALUASI MODEL**"""

# Mengambil model Random Forest yang sudah dilatih
best_model = models['Random Forest']

# Evaluasi eksplisit pada Test Set
y_pred_final = best_model.predict(X_test_processed)

rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
mae_final = mean_absolute_error(y_test, y_pred_final)
r2_final = r2_score(y_test, y_pred_final)

print("--- EVALUASI MODEL FINAL (RANDOM FOREST) ---")
print(f"RMSE : {rmse_final:.2f}")
print(f"MAE  : {mae_final:.2f}")
print(f"R²   : {r2_final:.2f}\n")

"""**FEATURE IMPORTANCE (KONTRIBUSI PREDIKSI)**"""

encoded_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(encoded_cat_cols)

importances = best_model.feature_importances_
feature_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
feature_imp.head(10).plot(kind='barh', color='teal')
plt.title('10 Fitur dengan Kontribusi Prediksi Terbesar (Random Forest)')
plt.xlabel('Nilai Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

"""**PREDIKSI DATA BARU (INFERENCE)**"""

sample_new_data = pd.DataFrame([{
    'Type of Food': 'Meat',
    'Number of Guests': 350,
    'Event Type': 'Corporate',
    'Quantity of Food': 450,
    'Storage Conditions': 'Refrigerated',
    'Purchase History': 'Regular',
    'Seasonality': 'All Seasons',
    'Preparation Method': 'Buffet',
    'Geographical Location': 'Urban',
    'Pricing': 'Moderate'
}])

sample_processed = preprocessor.transform(sample_new_data)
prediksi_terbuang = best_model.predict(sample_processed)

print(f"Estimasi Makanan Terbuang (Prediksi Model): {prediksi_terbuang[0]:.2f} kg")

"""**Ekspor model dan preprocessor ke file .pkl**"""

# Menyimpan model dan preprocessor
joblib.dump(best_model, 'food_wastage_model.pkl')
joblib.dump(preprocessor, 'food_wastage_preprocessor.pkl')

print("Model dan Preprocessor berhasil disimpan!")