import joblib
import pandas as pd

# Load artefak yang sudah disimpan
preprocessor = joblib.load('food_wastage_preprocessor.pkl')
model = joblib.load('food_wastage_model.pkl')

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

# Melakukan prediksi dengan model yang sudah di-load
predicted_wastage = model.predict(input_processed)

print("\n--- HASIL PREDIKSI ---")
print(f"Estimasi makanan terbuang (Wastage Food Amount): {predicted_wastage[0]:.2f} kg")
