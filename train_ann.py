import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🌾 ANN-BASED SMART AGRICULTURE - 24 INDIAN CROPS")
print("=" * 70)

# Create necessary folders
os.makedirs('dataset', exist_ok=True)
os.makedirs('ml_model', exist_ok=True)

# Generate enhanced dataset with 24 Indian crops
print("\n📊 Generating dataset with 24 Indian crops...")
np.random.seed(42)
n_samples = 10000  # Increased for better training with more crops

# Generate features with realistic distributions
temperature = np.random.normal(28, 6, n_samples).clip(10, 45)
humidity = np.random.normal(65, 15, n_samples).clip(30, 95)
rainfall = np.random.gamma(2.5, 60, n_samples).clip(0, 500)
soil_moisture = np.random.normal(45, 12, n_samples).clip(15, 80)
soil_ph = np.random.normal(6.5, 0.6, n_samples).clip(5.0, 8.5)

# List of 24 Indian crops with their optimal conditions
crop_database = [
    # Cereals
    {'name': 'Rice', 'optimal_temp': 28, 'temp_range': 10, 'optimal_humidity': 75, 'humidity_range': 20,
     'optimal_rainfall': 200, 'rainfall_range': 100, 'optimal_ph': 6.2, 'ph_range': 1.0, 'base_yield': 4.5,
     'category': 'Cereal', 'states': ['West Bengal', 'Uttar Pradesh', 'Punjab', 'Tamil Nadu', 'Andhra Pradesh'],
     'season': 'Kharif'},
    
    {'name': 'Wheat', 'optimal_temp': 22, 'temp_range': 8, 'optimal_humidity': 60, 'humidity_range': 15,
     'optimal_rainfall': 100, 'rainfall_range': 50, 'optimal_ph': 6.8, 'ph_range': 1.0, 'base_yield': 3.8,
     'category': 'Cereal', 'states': ['Uttar Pradesh', 'Punjab', 'Haryana', 'Madhya Pradesh', 'Rajasthan'],
     'season': 'Rabi'},
    
    {'name': 'Maize', 'optimal_temp': 30, 'temp_range': 8, 'optimal_humidity': 65, 'humidity_range': 15,
     'optimal_rainfall': 120, 'rainfall_range': 60, 'optimal_ph': 6.5, 'ph_range': 1.2, 'base_yield': 3.5,
     'category': 'Cereal', 'states': ['Karnataka', 'Madhya Pradesh', 'Bihar', 'Tamil Nadu', 'Telangana'],
     'season': 'Kharif/Rabi'},
    
    {'name': 'Barley', 'optimal_temp': 20, 'temp_range': 8, 'optimal_humidity': 55, 'humidity_range': 15,
     'optimal_rainfall': 70, 'rainfall_range': 35, 'optimal_ph': 7.0, 'ph_range': 1.0, 'base_yield': 2.5,
     'category': 'Cereal', 'states': ['Rajasthan', 'Uttar Pradesh', 'Madhya Pradesh', 'Haryana', 'Punjab'],
     'season': 'Rabi'},
    
    # Cash Crops
    {'name': 'Cotton', 'optimal_temp': 32, 'temp_range': 8, 'optimal_humidity': 55, 'humidity_range': 15,
     'optimal_rainfall': 80, 'rainfall_range': 50, 'optimal_ph': 7.2, 'ph_range': 1.2, 'base_yield': 2.8,
     'category': 'Cash Crop', 'states': ['Gujarat', 'Maharashtra', 'Telangana', 'Karnataka', 'Punjab'],
     'season': 'Kharif'},
    
    {'name': 'Sugarcane', 'optimal_temp': 28, 'temp_range': 8, 'optimal_humidity': 70, 'humidity_range': 15,
     'optimal_rainfall': 180, 'rainfall_range': 80, 'optimal_ph': 6.5, 'ph_range': 1.0, 'base_yield': 5.2,
     'category': 'Cash Crop', 'states': ['Uttar Pradesh', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Bihar'],
     'season': 'Annual'},
    
    {'name': 'Tobacco', 'optimal_temp': 26, 'temp_range': 8, 'optimal_humidity': 65, 'humidity_range': 15,
     'optimal_rainfall': 100, 'rainfall_range': 50, 'optimal_ph': 6.2, 'ph_range': 1.0, 'base_yield': 2.5,
     'category': 'Cash Crop', 'states': ['Gujarat', 'Andhra Pradesh', 'Karnataka', 'Uttar Pradesh', 'Maharashtra'],
     'season': 'Rabi'},
    
    # Oilseeds
    {'name': 'Groundnut', 'optimal_temp': 28, 'temp_range': 6, 'optimal_humidity': 60, 'humidity_range': 15,
     'optimal_rainfall': 100, 'rainfall_range': 50, 'optimal_ph': 6.2, 'ph_range': 1.0, 'base_yield': 2.2,
     'category': 'Oilseed', 'states': ['Gujarat', 'Andhra Pradesh', 'Tamil Nadu', 'Karnataka', 'Maharashtra'],
     'season': 'Kharif'},
    
    {'name': 'Soybean', 'optimal_temp': 28, 'temp_range': 6, 'optimal_humidity': 65, 'humidity_range': 15,
     'optimal_rainfall': 120, 'rainfall_range': 60, 'optimal_ph': 6.5, 'ph_range': 1.0, 'base_yield': 2.5,
     'category': 'Oilseed', 'states': ['Madhya Pradesh', 'Maharashtra', 'Rajasthan', 'Karnataka', 'Telangana'],
     'season': 'Kharif'},
    
    {'name': 'Mustard', 'optimal_temp': 20, 'temp_range': 8, 'optimal_humidity': 55, 'humidity_range': 15,
     'optimal_rainfall': 80, 'rainfall_range': 40, 'optimal_ph': 6.8, 'ph_range': 1.0, 'base_yield': 1.8,
     'category': 'Oilseed', 'states': ['Rajasthan', 'Uttar Pradesh', 'Haryana', 'Madhya Pradesh', 'Gujarat'],
     'season': 'Rabi'},
    
    {'name': 'Sunflower', 'optimal_temp': 26, 'temp_range': 8, 'optimal_humidity': 60, 'humidity_range': 15,
     'optimal_rainfall': 100, 'rainfall_range': 50, 'optimal_ph': 6.8, 'ph_range': 1.0, 'base_yield': 2.2,
     'category': 'Oilseed', 'states': ['Karnataka', 'Maharashtra', 'Andhra Pradesh', 'Tamil Nadu', 'Telangana'],
     'season': 'Kharif/Rabi'},
    
    # Vegetables
    {'name': 'Potato', 'optimal_temp': 22, 'temp_range': 6, 'optimal_humidity': 70, 'humidity_range': 15,
     'optimal_rainfall': 90, 'rainfall_range': 40, 'optimal_ph': 5.8, 'ph_range': 0.8, 'base_yield': 4.0,
     'category': 'Vegetable', 'states': ['Uttar Pradesh', 'West Bengal', 'Bihar', 'Punjab', 'Madhya Pradesh'],
     'season': 'Rabi'},
    
    {'name': 'Tomato', 'optimal_temp': 24, 'temp_range': 6, 'optimal_humidity': 65, 'humidity_range': 15,
     'optimal_rainfall': 100, 'rainfall_range': 50, 'optimal_ph': 6.2, 'ph_range': 0.8, 'base_yield': 3.5,
     'category': 'Vegetable', 'states': ['Andhra Pradesh', 'Karnataka', 'Maharashtra', 'Madhya Pradesh', 'Tamil Nadu'],
     'season': 'Rabi/Kharif'},
    
    {'name': 'Onion', 'optimal_temp': 22, 'temp_range': 6, 'optimal_humidity': 60, 'humidity_range': 15,
     'optimal_rainfall': 80, 'rainfall_range': 40, 'optimal_ph': 6.5, 'ph_range': 0.8, 'base_yield': 3.0,
     'category': 'Vegetable', 'states': ['Maharashtra', 'Karnataka', 'Madhya Pradesh', 'Gujarat', 'Bihar'],
     'season': 'Rabi'},
    
    # Spices
    {'name': 'Chilli', 'optimal_temp': 26, 'temp_range': 6, 'optimal_humidity': 65, 'humidity_range': 15,
     'optimal_rainfall': 90, 'rainfall_range': 40, 'optimal_ph': 6.5, 'ph_range': 0.8, 'base_yield': 2.5,
     'category': 'Spice', 'states': ['Andhra Pradesh', 'Telangana', 'Karnataka', 'Madhya Pradesh', 'Maharashtra'],
     'season': 'Kharif'},
    
    {'name': 'Turmeric', 'optimal_temp': 28, 'temp_range': 6, 'optimal_humidity': 75, 'humidity_range': 15,
     'optimal_rainfall': 150, 'rainfall_range': 60, 'optimal_ph': 6.0, 'ph_range': 1.0, 'base_yield': 3.2,
     'category': 'Spice', 'states': ['Telangana', 'Karnataka', 'Tamil Nadu', 'Maharashtra', 'Odisha'],
     'season': 'Kharif'},
    
    {'name': 'Ginger', 'optimal_temp': 26, 'temp_range': 6, 'optimal_humidity': 80, 'humidity_range': 15,
     'optimal_rainfall': 180, 'rainfall_range': 70, 'optimal_ph': 6.2, 'ph_range': 1.0, 'base_yield': 2.8,
     'category': 'Spice', 'states': ['Karnataka', 'Odisha', 'West Bengal', 'Madhya Pradesh', 'Maharashtra'],
     'season': 'Kharif'},
    
    # Fruits
    {'name': 'Banana', 'optimal_temp': 28, 'temp_range': 8, 'optimal_humidity': 75, 'humidity_range': 15,
     'optimal_rainfall': 160, 'rainfall_range': 70, 'optimal_ph': 6.2, 'ph_range': 1.0, 'base_yield': 5.0,
     'category': 'Fruit', 'states': ['Maharashtra', 'Tamil Nadu', 'Gujarat', 'Andhra Pradesh', 'Karnataka'],
     'season': 'Annual'},
    
    {'name': 'Mango', 'optimal_temp': 30, 'temp_range': 8, 'optimal_humidity': 60, 'humidity_range': 15,
     'optimal_rainfall': 120, 'rainfall_range': 60, 'optimal_ph': 6.5, 'ph_range': 1.0, 'base_yield': 4.5,
     'category': 'Fruit', 'states': ['Uttar Pradesh', 'Andhra Pradesh', 'Telangana', 'Karnataka', 'Maharashtra'],
     'season': 'Annual'},
    
    # Plantation Crops
    {'name': 'Coconut', 'optimal_temp': 30, 'temp_range': 8, 'optimal_humidity': 75, 'humidity_range': 15,
     'optimal_rainfall': 200, 'rainfall_range': 80, 'optimal_ph': 6.2, 'ph_range': 1.0, 'base_yield': 4.2,
     'category': 'Plantation', 'states': ['Kerala', 'Tamil Nadu', 'Karnataka', 'Andhra Pradesh', 'Maharashtra'],
     'season': 'Annual'},
    
    {'name': 'Tea', 'optimal_temp': 22, 'temp_range': 6, 'optimal_humidity': 80, 'humidity_range': 15,
     'optimal_rainfall': 250, 'rainfall_range': 100, 'optimal_ph': 5.2, 'ph_range': 1.0, 'base_yield': 3.5,
     'category': 'Plantation', 'states': ['Assam', 'West Bengal', 'Tamil Nadu', 'Kerala', 'Himachal Pradesh'],
     'season': 'Annual'},
    
    {'name': 'Coffee', 'optimal_temp': 24, 'temp_range': 6, 'optimal_humidity': 75, 'humidity_range': 15,
     'optimal_rainfall': 200, 'rainfall_range': 80, 'optimal_ph': 5.8, 'ph_range': 1.0, 'base_yield': 2.8,
     'category': 'Plantation', 'states': ['Karnataka', 'Kerala', 'Tamil Nadu', 'Andhra Pradesh', 'Odisha'],
     'season': 'Annual'},
    
    # Pulses & Others
    {'name': 'Pulses', 'optimal_temp': 26, 'temp_range': 8, 'optimal_humidity': 55, 'humidity_range': 15,
     'optimal_rainfall': 80, 'rainfall_range': 40, 'optimal_ph': 6.5, 'ph_range': 1.0, 'base_yield': 1.8,
     'category': 'Pulse', 'states': ['Madhya Pradesh', 'Maharashtra', 'Rajasthan', 'Uttar Pradesh', 'Karnataka'],
     'season': 'Rabi/Kharif'},
    
    {'name': 'Jute', 'optimal_temp': 30, 'temp_range': 8, 'optimal_humidity': 80, 'humidity_range': 15,
     'optimal_rainfall': 200, 'rainfall_range': 80, 'optimal_ph': 6.2, 'ph_range': 1.0, 'base_yield': 3.2,
     'category': 'Fibre', 'states': ['West Bengal', 'Bihar', 'Assam', 'Odisha', 'Meghalaya'],
     'season': 'Kharif'}
]

print(f"✅ Loaded {len(crop_database)} Indian crops")

crops = []
irrigation_levels = []
irrigation_amounts = []
yields = []
crop_categories = []
growing_states = []
growing_seasons = []

# Generate data for all crops
for i in range(n_samples):
    # Calculate suitability scores for each crop
    scores = {}
    for crop in crop_database:
        # Temperature score
        temp_diff = abs(temperature[i] - crop['optimal_temp'])
        temp_score = max(0, 1 - (temp_diff / crop['temp_range']))
        
        # Humidity score
        humidity_diff = abs(humidity[i] - crop['optimal_humidity'])
        humidity_score = max(0, 1 - (humidity_diff / crop['humidity_range']))
        
        # Rainfall score
        rainfall_diff = abs(rainfall[i] - crop['optimal_rainfall'])
        rainfall_score = max(0, 1 - (rainfall_diff / crop['rainfall_range']))
        
        # Soil pH score
        ph_diff = abs(soil_ph[i] - crop['optimal_ph'])
        ph_score = max(0, 1 - (ph_diff / crop['ph_range']))
        
        # Composite score with weights
        composite_score = (
            temp_score * 0.25 +
            humidity_score * 0.20 +
            rainfall_score * 0.25 +
            ph_score * 0.30
        )
        
        # Add random variation
        scores[crop['name']] = composite_score + np.random.normal(0, 0.05)
    
    # Select crop with highest score
    selected_crop_name = max(scores, key=scores.get)
    selected_crop = next(crop for crop in crop_database if crop['name'] == selected_crop_name)
    
    # Determine irrigation requirements based on rainfall
    if rainfall[i] < selected_crop['optimal_rainfall'] * 0.7:
        irrigation = 'High'
        irrigation_amount = np.random.uniform(
            selected_crop['optimal_rainfall'] * 0.8,
            selected_crop['optimal_rainfall'] * 1.2
        )
    elif rainfall[i] < selected_crop['optimal_rainfall']:
        irrigation = 'Medium'
        irrigation_amount = np.random.uniform(
            selected_crop['optimal_rainfall'] * 0.4,
            selected_crop['optimal_rainfall'] * 0.7
        )
    else:
        irrigation = 'Low'
        irrigation_amount = np.random.uniform(
            selected_crop['optimal_rainfall'] * 0.1,
            selected_crop['optimal_rainfall'] * 0.3
        )
    
    # Calculate yield based on environmental factors
    temp_factor = max(0.5, 1 - abs(temperature[i] - selected_crop['optimal_temp']) / 20)
    humidity_factor = max(0.5, 1 - abs(humidity[i] - selected_crop['optimal_humidity']) / 30)
    rainfall_factor = max(0.5, 1 - abs(rainfall[i] - selected_crop['optimal_rainfall']) / 150)
    ph_factor = max(0.5, 1 - abs(soil_ph[i] - selected_crop['optimal_ph']) / 2)
    irrigation_factor = 1.2 if irrigation == 'High' else 1.0 if irrigation == 'Medium' else 0.8
    
    final_yield = (
        selected_crop['base_yield'] * 
        temp_factor * 
        humidity_factor * 
        rainfall_factor * 
        ph_factor * 
        irrigation_factor
    )
    
    crops.append(selected_crop_name)
    irrigation_levels.append(irrigation)
    irrigation_amounts.append(round(irrigation_amount, 0))
    yields.append(max(0.5, min(8.0, final_yield)))
    crop_categories.append(selected_crop['category'])
    growing_states.append(', '.join(selected_crop['states'][:3]))  # Top 3 states
    growing_seasons.append(selected_crop['season'])

# Create DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'rainfall': rainfall,
    'soil_moisture': soil_moisture,
    'soil_ph': soil_ph,
    'crop': crops,
    'irrigation': irrigation_levels,
    'irrigation_amount': irrigation_amounts,
    'yield': yields,
    'crop_category': crop_categories,
    'growing_state': growing_states,
    'growing_season': growing_seasons
})

# Save dataset
df.to_csv('dataset/agriculture.csv', index=False)
print(f"✅ Dataset created: {len(df)} samples")
print(f"✅ Saved to: dataset/agriculture.csv")
print(f"\n📈 Dataset Statistics:")
print(f"   Total Crops: {len(df['crop'].unique())}")
print(f"   Crop List: {', '.join(sorted(df['crop'].unique()))}")
print(f"   Categories: {df['crop_category'].unique().tolist()}")
print(f"   Seasons: {df['growing_season'].unique().tolist()}")
print(f"   Irrigation levels: {df['irrigation'].unique().tolist()}")
print(f"   Irrigation amount range: {df['irrigation_amount'].min():.0f} - {df['irrigation_amount'].max():.0f} mm")
print(f"   Yield range: {df['yield'].min():.1f} - {df['yield'].max():.1f} tons/ha")

# Preprocessing
print("\n🔄 Preprocessing data...")

# Features
X = df[['temperature', 'humidity', 'rainfall', 'soil_moisture', 'soil_ph']].values

# Targets
y_crop = df['crop']
y_irrigation = df['irrigation']
y_irrigation_amount = df['irrigation_amount'].values
y_yield = df['yield'].values

# Encode categorical variables
label_encoders = {}

# Encode crop
le_crop = LabelEncoder()
y_crop_encoded = le_crop.fit_transform(y_crop)
label_encoders['crop'] = le_crop
print(f"✅ Encoded {len(le_crop.classes_)} crops")

# Encode irrigation
le_irrigation = LabelEncoder()
y_irrigation_encoded = le_irrigation.fit_transform(y_irrigation)
label_encoders['irrigation'] = le_irrigation

# Split data
X_train, X_test, y_crop_train, y_crop_test = train_test_split(
    X, y_crop_encoded, test_size=0.2, random_state=42
)
X_train, X_test, y_irr_train, y_irr_test = train_test_split(
    X, y_irrigation_encoded, test_size=0.2, random_state=42
)
X_train, X_test, y_amount_train, y_amount_test = train_test_split(
    X, y_irrigation_amount, test_size=0.2, random_state=42
)
X_train, X_test, y_yield_train, y_yield_test = train_test_split(
    X, y_yield, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Training samples: {X_train_scaled.shape[0]}")
print(f"✅ Testing samples: {X_test_scaled.shape[0]}")

# Train models with enhanced architecture
print("\n🧠 Training Enhanced Artificial Neural Networks for 24 Crops...")

# 1. Crop Prediction Model - Random Forest for better multi-class
print("\n📌 Training Crop Prediction Model...")
crop_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
crop_model.fit(X_train_scaled, y_crop_train)
crop_score = crop_model.score(X_test_scaled, y_crop_test)
crop_mae = mean_absolute_error(y_crop_test, crop_model.predict(X_test_scaled))
print(f"   ✅ Crop Model R² Score: {crop_score:.3f}")
print(f"   ✅ Crop Model MAE: {crop_mae:.3f}")

# 2. Irrigation Level Prediction Model
print("\n📌 Training Irrigation Level Prediction Model...")
irrigation_model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
irrigation_model.fit(X_train_scaled, y_irr_train)
irrigation_score = irrigation_model.score(X_test_scaled, y_irr_test)
irrigation_mae = mean_absolute_error(y_irr_test, irrigation_model.predict(X_test_scaled))
print(f"   ✅ Irrigation Model R² Score: {irrigation_score:.3f}")
print(f"   ✅ Irrigation Model MAE: {irrigation_mae:.3f}")

# 3. Irrigation Amount Prediction Model
print("\n📌 Training Irrigation Amount Prediction Model...")
amount_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    batch_size=64,
    random_state=42,
    early_stopping=True
)
amount_model.fit(X_train_scaled, y_amount_train)
amount_pred = amount_model.predict(X_test_scaled)
amount_mae = mean_absolute_error(y_amount_test, amount_pred)
amount_r2 = r2_score(y_amount_test, amount_pred)
print(f"   ✅ Amount Model R² Score: {amount_r2:.3f}")
print(f"   ✅ Amount Model MAE: {amount_mae:.1f} mm")

# 4. Yield Prediction Model
print("\n📌 Training Yield Prediction Model...")
yield_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    batch_size=64,
    random_state=42,
    early_stopping=True
)
yield_model.fit(X_train_scaled, y_yield_train)
yield_pred = yield_model.predict(X_test_scaled)
yield_mae = mean_absolute_error(y_yield_test, yield_pred)
yield_r2 = r2_score(y_yield_test, yield_pred)
print(f"   ✅ Yield Model R² Score: {yield_r2:.3f}")
print(f"   ✅ Yield Model MAE: {yield_mae:.3f} tons/ha")

# Save models
print("\n💾 Saving trained models...")

joblib.dump(crop_model, 'ml_model/crop_model.pkl')
joblib.dump(irrigation_model, 'ml_model/irrigation_model.pkl')
joblib.dump(amount_model, 'ml_model/irrigation_amount_model.pkl')
joblib.dump(yield_model, 'ml_model/yield_model.pkl')
joblib.dump(scaler, 'ml_model/scaler.pkl')
joblib.dump(label_encoders, 'ml_model/label_encoders.pkl')

print("✅ Models saved to ml_model/ folder:")
print("   - crop_model.pkl (Random Forest)")
print("   - irrigation_model.pkl (Gradient Boosting)")
print("   - irrigation_amount_model.pkl (Deep ANN)")
print("   - yield_model.pkl (Deep ANN)")
print("   - scaler.pkl")
print("   - label_encoders.pkl")

# Test predictions with diverse crops
print("\n🎯 Testing Enhanced Predictions for 24 Crops...")
test_cases = [
    [25, 70, 160, 50, 6.2],  # Rice
    [22, 60, 100, 45, 6.8],  # Wheat
    [32, 55, 50, 35, 7.2],   # Cotton
    [30, 60, 120, 40, 6.5],  # Maize
    [28, 70, 180, 55, 6.5],  # Sugarcane
    [28, 60, 100, 45, 6.2],  # Groundnut
    [20, 55, 80, 40, 6.8],   # Mustard
    [22, 70, 90, 50, 5.8],   # Potato
    [30, 75, 200, 60, 6.2],  # Coconut
    [24, 80, 250, 65, 5.2],  # Tea
]

for i, test_case in enumerate(test_cases, 1):
    features = np.array([test_case])
    features_scaled = scaler.transform(features)
    
    # Predict
    crop_pred = int(round(crop_model.predict(features_scaled)[0]))
    irr_pred = int(round(irrigation_model.predict(features_scaled)[0]))
    amount_pred = amount_model.predict(features_scaled)[0]
    yield_pred = yield_model.predict(features_scaled)[0]
    
    # Decode
    crop_name = label_encoders['crop'].inverse_transform([crop_pred])[0]
    irr_name = label_encoders['irrigation'].inverse_transform([irr_pred])[0]
    
    # Find crop details
    crop_details = next((c for c in crop_database if c['name'] == crop_name), None)
    
    print(f"\n📋 Test Case {i}:")
    print(f"   Input: {test_case[0]}°C, {test_case[1]}% humidity, {test_case[2]}mm rain, {test_case[3]}% moisture, pH {test_case[4]}")
    print(f"   Output: 🌾 {crop_name}")
    if crop_details:
        print(f"           📊 Category: {crop_details['category']}")
        print(f"           📍 States: {', '.join(crop_details['states'][:3])}")
        print(f"           📅 Season: {crop_details['season']}")
    print(f"           💧 {irr_name} irrigation ({amount_pred:.0f} mm per season)")
    print(f"           ⚖️ Expected Yield: {yield_pred:.2f} tons/ha")

print("\n" + "=" * 70)
print("✅ ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
print(f"📊 Now supporting {len(crop_database)} Indian crops")
print("=" * 70)
print("\n🚀 Next Step: Run 'python app.py' to start the web application")
print("=" * 70)