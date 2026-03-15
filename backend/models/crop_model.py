import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os
import pickle

CROP_INFO = {
    "Rice": {
        "kannada": "ಭತ್ತ",
        "fertilizer": "Apply Urea (50kg/acre), DAP (25kg/acre), MOP (20kg/acre). Split N application in 3 doses.",
        "description": "Ideal for high nitrogen, moderate pH soils with good moisture retention.",
        "icon": "🌾"
    },
    "Wheat": {
        "kannada": "ಗೋಧಿ",
        "fertilizer": "Apply NPK 120:60:40 kg/ha. Zinc sulfate 25kg/ha if deficient.",
        "description": "Best in well-drained loamy soils with moderate organic carbon.",
        "icon": "🌾"
    },
    "Maize": {
        "kannada": "ಮೆಕ್ಕೆ ಜೋಳ",
        "fertilizer": "NPK 150:75:75 kg/ha. Apply Boron for better yield.",
        "description": "Requires good potassium levels and moderate moisture for optimal growth.",
        "icon": "🌽"
    },
    "Sugarcane": {
        "kannada": "ಕಬ್ಬು",
        "fertilizer": "NPK 250:100:120 kg/ha. Apply FYM 25 tons/ha before planting.",
        "description": "Thrives in high potassium, neutral pH soils with abundant moisture.",
        "icon": "🎋"
    },
    "Cotton": {
        "kannada": "ಹತ್ತಿ",
        "fertilizer": "NPK 120:60:60 kg/ha. Apply micronutrients Zinc and Boron.",
        "description": "Grows well in deep black cotton soils with moderate nitrogen.",
        "icon": "🌿"
    },
    "Groundnut": {
        "kannada": "ಶೇಂಗಾ",
        "fertilizer": "NPK 25:50:75 kg/ha. Apply Gypsum 400kg/ha for calcium and sulfur.",
        "description": "Prefers sandy loam soils with good phosphorus and calcium levels.",
        "icon": "🥜"
    },
    "Jowar": {
        "kannada": "ಜೋಳ",
        "fertilizer": "NPK 80:40:40 kg/ha. Organic manure 5 tons/ha recommended.",
        "description": "Drought-tolerant crop suitable for medium to low moisture soils.",
        "icon": "🌾"
    },
    "Ragi": {
        "kannada": "ರಾಗಿ",
        "fertilizer": "NPK 60:30:30 kg/ha. Well decomposed FYM 10 tons/ha.",
        "description": "Hardy crop that grows in poor soils with low to moderate fertility.",
        "icon": "🌾"
    },
    "Sunflower": {
        "kannada": "ಸೂರ್ಯಕಾಂತಿ",
        "fertilizer": "NPK 90:60:60 kg/ha. Apply Boron 1kg/ha for better seed set.",
        "description": "Performs well in well-drained soils with good organic matter content.",
        "icon": "🌻"
    },
    "Tomato": {
        "kannada": "ಟೊಮೆಟೊ",
        "fertilizer": "NPK 120:80:100 kg/ha. Calcium and Magnesium supplements needed.",
        "description": "Requires fertile, well-drained soils with high organic carbon.",
        "icon": "🍅"
    },
    "Onion": {
        "kannada": "ಈರುಳ್ಳಿ",
        "fertilizer": "NPK 100:50:100 kg/ha. Sulfur 30kg/ha improves flavor and yield.",
        "description": "Best in sandy loam to clay loam soils with good potassium.",
        "icon": "🧅"
    },
    "Chilli": {
        "kannada": "ಮೆಣಸಿನಕಾಯಿ",
        "fertilizer": "NPK 100:60:80 kg/ha. Calcium nitrate spray during fruiting.",
        "description": "Grows well in warm, well-drained soils with moderate fertility.",
        "icon": "🌶️"
    }
}

def generate_training_data(n_samples=3000):
    np.random.seed(42)
    data = []
    labels = []
    
    crop_params = {
        "Rice":      {"ph": (5.5, 7.0), "n": (80, 140), "p": (15, 35), "k": (100, 160), "oc": (0.15, 0.40), "temp": (22, 35), "moist": (60, 90)},
        "Wheat":     {"ph": (6.0, 7.5), "n": (60, 120), "p": (20, 40), "k": (80, 140), "oc": (0.18, 0.45), "temp": (15, 25), "moist": (40, 70)},
        "Maize":     {"ph": (5.8, 7.2), "n": (70, 130), "p": (18, 38), "k": (110, 170), "oc": (0.20, 0.50), "temp": (20, 32), "moist": (50, 75)},
        "Sugarcane": {"ph": (6.0, 7.5), "n": (90, 160), "p": (20, 45), "k": (120, 200), "oc": (0.25, 0.60), "temp": (25, 38), "moist": (65, 90)},
        "Cotton":    {"ph": (6.5, 8.0), "n": (60, 110), "p": (15, 35), "k": (90, 150), "oc": (0.10, 0.35), "temp": (25, 40), "moist": (30, 60)},
        "Groundnut": {"ph": (5.5, 7.0), "n": (20, 60), "p": (30, 60), "k": (80, 130), "oc": (0.15, 0.40), "temp": (25, 38), "moist": (40, 65)},
        "Jowar":     {"ph": (6.5, 8.0), "n": (40, 90), "p": (10, 30), "k": (70, 120), "oc": (0.10, 0.30), "temp": (25, 40), "moist": (25, 55)},
        "Ragi":      {"ph": (5.0, 6.5), "n": (30, 80), "p": (10, 28), "k": (50, 100), "oc": (0.08, 0.28), "temp": (20, 35), "moist": (25, 55)},
        "Sunflower": {"ph": (6.0, 7.5), "n": (50, 100), "p": (20, 45), "k": (80, 130), "oc": (0.15, 0.40), "temp": (22, 35), "moist": (35, 60)},
        "Tomato":    {"ph": (5.8, 7.0), "n": (80, 130), "p": (25, 55), "k": (100, 160), "oc": (0.25, 0.60), "temp": (18, 30), "moist": (50, 75)},
        "Onion":     {"ph": (6.0, 7.5), "n": (70, 120), "p": (20, 50), "k": (100, 160), "oc": (0.15, 0.40), "temp": (18, 28), "moist": (40, 65)},
        "Chilli":    {"ph": (6.0, 7.5), "n": (60, 110), "p": (20, 45), "k": (80, 140), "oc": (0.20, 0.50), "temp": (20, 35), "moist": (40, 65)},
    }
    
    crops = list(crop_params.keys())
    per_crop = n_samples // len(crops)
    
    for crop in crops:
        p = crop_params[crop]
        for _ in range(per_crop):
            ph = np.random.uniform(*p["ph"])
            ec = np.random.uniform(0.1, 0.5)
            oc = np.random.uniform(*p["oc"])
            n = np.random.uniform(*p["n"])
            p2o5 = np.random.uniform(*p["p"])
            k2o = np.random.uniform(*p["k"])
            cec = np.random.uniform(8, 25)
            temp = np.random.uniform(*p["temp"])
            moist = np.random.uniform(*p["moist"])
            noise = np.random.normal(0, 0.05)
            row = [
                ph + noise, ec, oc + noise*0.01, n + noise*5,
                p2o5 + noise*2, k2o + noise*5, cec,
                ph - 0.1 + noise, ec + 0.05, oc + 0.02,
                n + 10 + noise*3, p2o5 + 2 + noise, k2o - 8 + noise*3,
                cec - 2, temp, moist
            ]
            data.append(row)
            labels.append(crop)
    
    return np.array(data), np.array(labels)


class CropRecommender:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "crop_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                saved = pickle.load(f)
                self.model = saved["model"]
                self.scaler = saved["scaler"]
                self.classes = saved["classes"]
        else:
            self._train_and_save(model_path)

    def _train_and_save(self, model_path):
        print("Training crop recommendation model...")
        X, y = generate_training_data(3600)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
            min_samples_split=5, class_weight="balanced"
        )
        self.model.fit(X_train_scaled, y_train)
        self.classes = self.model.classes_
        acc = self.model.score(X_test_scaled, y_test)
        print(f"Model accuracy: {acc:.2%}")
        
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "classes": self.classes}, f)

    def predict(self, data: dict):
        features = [
            data["ph_d1"], data["ec_d1"], data["oc_d1"], data["n_d1"],
            data["p2o5_d1"], data["k2o_d1"], data["cec_d1"],
            data["ph_d2"], data["ec_d2"], data["oc_d2"], data["n_d2"],
            data["p2o5_d2"], data["k2o_d2"], data["cec_d2"],
            data["temperature"], data["moisture"]
        ]
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        
        best_crop = self.classes[top3_idx[0]]
        best_conf = float(proba[top3_idx[0]]) * 100
        alt1 = self.classes[top3_idx[1]]
        alt1_conf = float(proba[top3_idx[1]]) * 100
        alt2 = self.classes[top3_idx[2]]
        alt2_conf = float(proba[top3_idx[2]]) * 100
        
        info = CROP_INFO.get(best_crop, {})
        
        return {
            "best_crop": best_crop,
            "best_crop_kannada": info.get("kannada", ""),
            "confidence": round(best_conf, 1),
            "icon": info.get("icon", "🌱"),
            "description": info.get("description", ""),
            "fertilizer": info.get("fertilizer", ""),
            "alternatives": [
                {"crop": alt1, "confidence": round(alt1_conf, 1), "kannada": CROP_INFO.get(alt1, {}).get("kannada", ""), "icon": CROP_INFO.get(alt1, {}).get("icon", "🌿")},
                {"crop": alt2, "confidence": round(alt2_conf, 1), "kannada": CROP_INFO.get(alt2, {}).get("kannada", ""), "icon": CROP_INFO.get(alt2, {}).get("icon", "🌿")},
            ],
            "soil_analysis": _analyze_soil(data)
        }


def _analyze_soil(data):
    notes = []
    avg_ph = (data["ph_d1"] + data["ph_d2"]) / 2
    avg_n = (data["n_d1"] + data["n_d2"]) / 2
    avg_p = (data["p2o5_d1"] + data["p2o5_d2"]) / 2
    avg_k = (data["k2o_d1"] + data["k2o_d2"]) / 2
    
    if avg_ph < 6.0:
        notes.append("Acidic soil — apply lime to raise pH")
    elif avg_ph > 8.0:
        notes.append("Alkaline soil — apply gypsum or sulfur")
    else:
        notes.append("Optimal pH range")
    
    if avg_n < 50:
        notes.append("Low nitrogen — apply urea or organic manure")
    elif avg_n > 150:
        notes.append("High nitrogen — reduce N fertilizer")
    else:
        notes.append("Adequate nitrogen levels")
    
    if avg_p < 15:
        notes.append("Phosphorus deficient — apply DAP")
    else:
        notes.append("Good phosphorus content")
    
    if avg_k < 80:
        notes.append("Low potassium — apply MOP")
    else:
        notes.append("Good potassium levels")
    
    return notes
