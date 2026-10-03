import joblib
import pandas as pd

# Load artefak yang sudah disimpan
preprocessor = joblib.load('food_wastage_preprocessor.pkl')
model = joblib.load('food_wastage_model.pkl')

# Input data baru
sample_new_data = pd.DataFrame([{
    'Type of Food': 'Meat',
    'Number of Guests': 300,
    'Event Type': 'Corporate',
    'Quantity of Food': 400,
    'Storage Conditions': 'Refrigerated',
    'Purchase History': 'Regular',
    'Seasonality': 'All Seasons',
    'Preparation Method': 'Buffet',
    'Geographical Location': 'Urban',
    'Pricing': 'Moderate'
}])

# Prediksi
sample_processed = preprocessor.transform(sample_new_data)
prediksi = model.predict(sample_processed)

print(f"Estimasi Makanan Terbuang: {prediksi[0]:.2f} kg")