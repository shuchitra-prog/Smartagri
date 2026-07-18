import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# Ensure folders exist
os.makedirs("ml_model", exist_ok=True)
os.makedirs("dataset", exist_ok=True)


def create_sample_dataset():

    print("📝 Creating sample dataset...")

    data = []

    crops = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Barley", "Potato"]
    irrig = ["Low", "Medium", "High"]

    for _ in range(300):

        temp = np.random.uniform(18, 40)
        hum = np.random.uniform(40, 85)
        rain = np.random.uniform(20, 200)
        soil = np.random.uniform(20, 70)
        ph = np.random.uniform(5.5, 8)

        crop = np.random.choice(crops)
        irrigation = np.random.choice(irrig)
        yield_val = round(np.random.uniform(1.5, 6.0), 2)

        data.append([temp, hum, rain, soil, ph, crop, irrigation, yield_val])

    df = pd.DataFrame(data, columns=[
        "temperature","humidity","rainfall","soil_moisture","soil_ph",
        "crop","irrigation","yield"
    ])

    df.to_csv("dataset/agriculture.csv", index=False)

    print("✅ Dataset created successfully.")


def create_and_train_ann():

    if not os.path.exists("dataset/agriculture.csv"):
        create_sample_dataset()

    print("📊 Loading dataset...")

    df = pd.read_csv("dataset/agriculture.csv")

    X = df[["temperature","humidity","rainfall","soil_moisture","soil_ph"]]

    crop_enc = LabelEncoder()
    irr_enc = LabelEncoder()

    df["crop"] = crop_enc.fit_transform(df["crop"])
    df["irrigation"] = irr_enc.fit_transform(df["irrigation"])

    Y = np.column_stack((df["crop"], df["irrigation"], df["yield"]))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump({"crop":crop_enc,"irr":irr_enc}, "ml_model/label_encoders.pkl")
    joblib.dump(scaler, "ml_model/scaler.pkl")

    print("🧠 Building ANN...")

    model = Sequential([
        Dense(32, activation="relu", input_shape=(5,)),
        Dense(32, activation="relu"),
        Dense(3)
    ])

    model.compile(optimizer="adam", loss="mse")

    print("🚀 Training ANN...")

    model.fit(X, Y, epochs=120, batch_size=16, verbose=1)

    model.save("ml_model/ann_model.keras")

    print("\n✅ ANN TRAINING COMPLETE")
    print("📁 Saved:")
    print(" - ml_model/ann_model.keras")
    print(" - ml_model/label_encoders.pkl")
    print(" - ml_model/scaler.pkl")

    return model


def test_prediction():

    model = load_model("ml_model/ann_model.keras", compile=False)

    enc = joblib.load("ml_model/label_encoders.pkl")
    scaler = joblib.load("ml_model/scaler.pkl")

    sample = np.array([[28,65,120,45,6.5]])
    sample = scaler.transform(sample)

    pred = model.predict(sample)[0]

    crop = enc["crop"].inverse_transform([int(round(pred[0]))])[0]
    irr = enc["irr"].inverse_transform([int(round(pred[1]))])[0]
    yld = round(float(pred[2]),2)

    print("\n🔮 SAMPLE PREDICTION")
    print("Crop:",crop)
    print("Irrigation:",irr)
    print("Yield:",yld,"tons")


if __name__ == "__main__":

    print("="*50)
    print("🤖 ANN SMART AGRICULTURE TRAINING")
    print("="*50)

    create_and_train_ann()
    test_prediction()
