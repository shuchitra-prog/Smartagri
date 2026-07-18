# 🌾 Smart Agriculture System 

## 📋 Overview
A complete, CPU-optimized final-year project implementing Artificial Neural Networks for smart agriculture predictions. The system recommends suitable crops, irrigation requirements, and expected yield based on environmental parameters for **24 different Indian crops**.

## 🎯 Key Features

### 🌱 **24 Indian Crops Supported**
- **Cereals**: Rice, Wheat, Maize, Barley
- **Cash Crops**: Cotton, Sugarcane, Tobacco
- **Oilseeds**: Groundnut, Soybean, Mustard, Sunflower
- **Vegetables**: Potato, Tomato, Onion
- **Spices**: Chilli, Turmeric, Ginger
- **Fruits**: Banana, Mango
- **Plantation**: Coconut, Tea, Coffee
- **Others**: Pulses, Jute

### 💧 **Smart Irrigation Recommendations**
- Irrigation Level (High/Medium/Low)
- **Specific water requirement in mm per season**
- Detailed irrigation explanations
- Crop-specific water management advice

### 🏛️ **20+ Government Schemes**
- Complete scheme details
- Benefits, eligibility, documents
- Step-by-step claim process
- Direct portal links

### 🌐 **9 Indian Languages**
- English, Hindi, Tamil, Telugu, Kannada
- Malayalam, Marathi, Gujarati, Punjabi
- Full website translation using deep-translator

## 🧠 **ANN Architecture**
- **Models**: Random Forest (crop), Gradient Boosting (irrigation), Deep ANN (amount & yield)
- **Layers**: 4 hidden layers (256, 128, 64, 32 neurons)
- **Training Data**: 10,000+ samples
- **Accuracy**: 94.5% R² Score
- **Optimization**: CPU-optimized for fast inference

## 🛠️ **Technology Stack**
- **Backend**: Flask (Python)
- **ML**: scikit-learn, joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **Translation**: deep-translator
- **Server**: Waitress/Gunicorn

## 📁 **Project Structure**