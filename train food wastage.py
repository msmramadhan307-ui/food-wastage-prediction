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
# Input data baru secara interaktif dengan otomatis konversi ke Title Case
print("--- INPUT DATA UNTUK PREDIKSI WASTAGE --- ")

type_of_food = (input("Type of Food (e.g., Meat, Vegetables, Baked Goods): ") or "Meat").strip().title()
number_of_guests = int(input("Number of Guests (e.g., 350): ") or 350)
event_type = (input("Event Type (e.g., Corporate, Birthday): ") or "Corporate").strip().title()
quantity_of_food = float(input("Quantity of Food (e.g., 450): ") or 450)
storage_conditions = (input("Storage Conditions (e.g., Refrigerated, Room Temperature): ") or "Refrigerated").strip().title()
purchase_history = (input("Purchase History (e.g., Regular, Occasional): ") or "Regular").strip().title()
seasonality = (input("Seasonality (e.g., All Seasons, Winter, Summer): ") or "All Seasons").strip().title()
preparation_method = (input("Preparation Method (e.g., Buffet, Finger Food): ") or "Buffet").strip().title()
geographical_location = (input("Geographical Location (e.g., Urban, Suburban, Rural): ") or "Urban").strip().title()
pricing = (input("Pricing (e.g., Moderate, Low, High): ") or "Moderate").strip().title()

# Memasukkan input ke dalam DataFrame
input_data = pd.DataFrame([{
    'Type of Food': type_of_food,
    'Number of Guests': number_of_guests,
    'Event Type': event_type,
    'Quantity of Food': quantity_of_food,
    'Storage Conditions': storage_conditions,
    'Purchase History': purchase_history,
    'Seasonality': seasonality,
    'Preparation Method': preparation_method,
    'Geographical Location': geographical_location,
    'Pricing': pricing
}])

# Melakukan transformasi data menggunakan preprocessor yang sudah dilatih
input_processed = preprocessor.transform(input_data)

# Melakukan prediksi dengan model terbaik (best_model)
predicted_wastage = best_model.predict(input_processed)

print("\n--- HASIL PREDIKSI ---")
print(f"Estimasi makanan terbuang (Wastage Food Amount): {predicted_wastage[0]:.2f} kg")

"""**Ekspor model dan preprocessor ke file .pkl**"""

# Menyimpan model dan preprocessor
joblib.dump(best_model, 'food_wastage_model.pkl')
joblib.dump(preprocessor, 'food_wastage_preprocessor.pkl')

print("Model dan Preprocessor berhasil disimpan!")
