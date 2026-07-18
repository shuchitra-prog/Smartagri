from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
import os
from tensorflow.keras.models import load_model
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)

# Load ANN + utilities
model = load_model("ml_model/ann_model.keras", compile=False)
encoders = joblib.load("ml_model/label_encoders.pkl")
scaler = joblib.load("ml_model/scaler.pkl")

# Load government schemes
with open("data/govt_schemes.json", "r", encoding="utf-8") as f:
    schemes = json.load(f)

# Load translations
with open("data/translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

# ANN Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        vals = [
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["rainfall"]),
            float(data["soil_moisture"]),
            float(data["soil_ph"])
        ]

        sample = scaler.transform([vals])
        pred = model.predict(sample)[0]

        crop = encoders["crop"].inverse_transform([int(round(pred[0]))])[0]
        irrigation = encoders["irr"].inverse_transform([int(round(pred[1]))])[0]
        yield_val = round(float(pred[2]),2)

        return jsonify({
            "success": True,
            "crop": crop,
            "irrigation": irrigation,
            "yield": yield_val
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Translate the entire interface
@app.route("/translate_ui", methods=["POST"])
def translate_ui():
    try:
        data = request.json
        target_lang = data.get("lang", "en")
        
        # Get translations for the requested language
        lang_translations = translations.get(target_lang, translations.get("en", {}))
        
        return jsonify({
            "success": True,
            "translations": lang_translations,
            "language": target_lang
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Translate individual text (for dynamic content)
@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.json
        text = data["text"]
        lang = data["lang"]
        
        translated = GoogleTranslator(source="auto", target=lang).translate(text)
        return jsonify({"success": True, "translated": translated})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Govt Schemes with translation
@app.route("/schemes")
def govt():
    lang = request.args.get("lang", "en")
    
    if lang == "en":
        return jsonify(schemes)
    
    # Translate schemes if needed
    translated_schemes = []
    for scheme in schemes:
        try:
            translated_scheme = scheme.copy()
            # Translate title and description
            if lang != "en":
                translated_scheme["title"] = GoogleTranslator(source="auto", target=lang).translate(scheme["title"])
                translated_scheme["description"] = GoogleTranslator(source="auto", target=lang).translate(scheme["description"])
            translated_schemes.append(translated_scheme)
        except:
            translated_schemes.append(scheme)
    
    return jsonify(translated_schemes)

# ANN explanation with translation
@app.route("/ann_explanation")
def ann_explain():
    lang = request.args.get("lang", "en")
    
    explanation = {
        "en": "ANN works like a human brain. It learns from past farming data and predicts best crop, irrigation and yield.",
        "hi": "ANN मानव मस्तिष्क की तरह काम करता है। यह पिछले कृषि डेटा से सीखता है और सर्वोत्तम फसल, सिंचाई और उपज की भविष्यवाणी करता है।",
        "ta": "ANN ஒரு மனித மூளையைப் போல வேலை செய்கிறது. இது கடந்த விவசாயத் தரவுகளிலிருந்து கற்றுக்கொள்கிறது மற்றும் சிறந்த பயிர், பாசனம் மற்றும் மகசூலை கணிக்கிறது.",
        "te": "ANN మానవ మెదడు వలె పనిచేస్తుంది. ఇది గత వ్యవసాయ డేటా నుండి నేర్చుకుంటుంది మరియు ఉత్తమ పంట, నీటిపారుదల మరియు దిగుబడిని అంచనా వేస్తుంది."
    }
    
    return jsonify({
        "text": explanation.get(lang, explanation["en"])
    })

# Get available languages
@app.route("/languages")
def get_languages():
    languages = [
        {"code": "en", "name": "English"},
        {"code": "hi", "name": "हिंदी"},
        {"code": "ta", "name": "தமிழ்"},
        {"code": "te", "name": "తెలుగు"},
        {"code": "ml", "name": "മലയാളം"},
        {"code": "kn", "name": "ಕನ್ನಡ"},
        {"code": "bn", "name": "বাংলা"},
        {"code": "gu", "name": "ગુજરાતી"},
        {"code": "mr", "name": "मराठी"},
        {"code": "pa", "name": "ਪੰਜਾਬੀ"}
    ]
    return jsonify(languages)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("ml_model", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Create translations.json if it doesn't exist
    translations_path = "data/translations.json"
    if not os.path.exists(translations_path):
        default_translations = {
            "en": {
                "app_title": "Smart Agriculture System",
                "temperature": "Temperature (°C)",
                "humidity": "Humidity (%)",
                "rainfall": "Rainfall (mm)",
                "soil_moisture": "Soil Moisture (%)",
                "soil_ph": "Soil pH",
                "predict": "Predict",
                "reset": "Reset",
                "loading": "Loading...",
                "crop_recommendation": "Crop Recommendation",
                "irrigation_method": "Irrigation Method",
                "expected_yield": "Expected Yield (tons/ha)",
                "select_language": "Select Language",
                "govt_schemes": "Government Schemes",
                "ann_explanation": "How ANN Works",
                "error": "Error",
                "success": "Success",
                "prediction_results": "Prediction Results",
                "enter_details": "Enter Farming Details",
                "features": "Features",
                "prediction": "Prediction",
                "value": "Value",
                "scheme_title": "Title",
                "scheme_desc": "Description",
                "scheme_link": "Learn More",
                "close": "Close"
            },
            "hi": {
                "app_title": "स्मार्ट कृषि प्रणाली",
                "temperature": "तापमान (°C)",
                "humidity": "आर्द्रता (%)",
                "rainfall": "वर्षा (मिमी)",
                "soil_moisture": "मृदा आर्द्रता (%)",
                "soil_ph": "मृदा पीएच",
                "predict": "भविष्यवाणी करें",
                "reset": "रीसेट करें",
                "loading": "लोड हो रहा है...",
                "crop_recommendation": "फसल सिफारिश",
                "irrigation_method": "सिंचाई विधि",
                "expected_yield": "अपेक्षित उपज (टन/हेक्टेयर)",
                "select_language": "भाषा चुनें",
                "govt_schemes": "सरकारी योजनाएं",
                "ann_explanation": "ANN कैसे काम करता है",
                "error": "त्रुटि",
                "success": "सफलता",
                "prediction_results": "भविष्यवाणी परिणाम",
                "enter_details": "कृषि विवरण दर्ज करें",
                "features": "विशेषताएं",
                "prediction": "भविष्यवाणी",
                "value": "मूल्य",
                "scheme_title": "शीर्षक",
                "scheme_desc": "विवरण",
                "scheme_link": "अधिक जानें",
                "close": "बंद करें"
            }
            # Add more languages as needed
        }
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(default_translations, f, ensure_ascii=False, indent=2)

    print("="*50)
    print("🌱  SMART AGRICULTURE SERVER")
    print("="*50)
    print("🚀 http://127.0.0.1:5000")
    print("="*50)

    app.run(debug=True)