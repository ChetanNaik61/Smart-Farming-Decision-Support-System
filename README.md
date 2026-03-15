# 🌿 Smart Farming Decision Support System
### AI-Powered Crop Recommendation with Kannada Voice Assistant

A final year engineering project demonstrating Machine Learning, FastAPI, and regional language AI for precision agriculture.

---

## 📁 Project Structure

```
smart-farming/
├── backend/
│   ├── main.py                  # FastAPI application entry point
│   ├── requirements.txt         # Python dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   └── crop_model.py        # Random Forest ML model
│   └── routes/
│       ├── __init__.py
│       └── chat.py              # Kannada chatbot API route
└── frontend/
    └── templates/
        ├── index.html           # Home page
        ├── analyze.html         # Soil analysis + results page
        └── assistant.html       # Kannada voice assistant page
```

---

## 🚀 Features

- **Soil Lab Report Input** — Enter pH, EC, OC, N, P₂O₅, K₂O, CEC at two soil depths
- **ML Crop Recommendation** — Random Forest model trained on 3600+ samples across 12 crops
- **Confidence Score** — Visual ring indicator showing prediction confidence
- **Fertilizer Guidance** — Specific NPK ratios per recommended crop
- **Soil Health Analysis** — Identifies deficiencies and improvement steps
- **Kannada Voice Chatbot** — Type or speak in Kannada, get Kannada voice response
- **12 Crops Supported** — Rice, Wheat, Maize, Sugarcane, Cotton, Groundnut, Jowar, Ragi, Sunflower, Tomato, Onion, Chilli

---

## 🪟 Windows Setup Instructions

### Prerequisites
1. **Python 3.10+** — Download from https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation
2. **Git** (optional) — https://git-scm.com/download/win

### Step-by-Step

```cmd
REM 1. Open Command Prompt (Win + R → cmd)

REM 2. Navigate to project folder
cd path\to\smart-farming

REM 3. Create virtual environment
python -m venv venv

REM 4. Activate virtual environment
venv\Scripts\activate

REM 5. Install dependencies
pip install fastapi uvicorn[standard] python-multipart pydantic scikit-learn numpy pandas httpx

REM 6. Start the backend server
cd backend
python main.py

REM 7. Open your browser and visit:
REM    http://localhost:8000
```

### Windows PowerShell (Alternative)
```powershell
# Activate virtual environment in PowerShell
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🐧 Linux / Ubuntu Setup Instructions

### Prerequisites
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Verify installation
python3 --version
pip3 --version
```

### Step-by-Step

```bash
# 1. Navigate to project folder
cd ~/smart-farming

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install fastapi uvicorn[standard] python-multipart pydantic scikit-learn numpy pandas httpx

# 5. Start the backend server
cd backend
python main.py

# 6. Open browser and visit:
#    http://localhost:8000
```

### Run in Background (Linux)
```bash
# Run server in background
nohup python main.py &

# Or use screen
screen -S smartfarm
python main.py
# Ctrl+A then D to detach
```

---

## 🌐 Pages

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Home page |
| `http://localhost:8000/analyze` | Soil analysis & crop recommendation |
| `http://localhost:8000/assistant` | Kannada voice assistant |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/recommend` | Get crop recommendation from soil data |
| POST | `/api/chat` | Kannada chatbot response |

### Example API Request — Crop Recommendation

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "ph_d1": 7.83, "ec_d1": 0.14, "oc_d1": 0.21,
    "n_d1": 100.35, "p2o5_d1": 18.64, "k2o_d1": 134.4, "cec_d1": 13.45,
    "ph_d2": 7.60, "ec_d2": 0.21, "oc_d2": 0.24,
    "n_d2": 112.89, "p2o5_d2": 21.18, "k2o_d2": 123.6, "cec_d2": 10.30,
    "temperature": 28, "moisture": 55
  }'
```

### Example Response

```json
{
  "best_crop": "Jowar",
  "best_crop_kannada": "ಜೋಳ",
  "confidence": 84.3,
  "icon": "🌾",
  "description": "Drought-tolerant crop suitable for medium to low moisture soils.",
  "fertilizer": "NPK 80:40:40 kg/ha. Organic manure 5 tons/ha recommended.",
  "alternatives": [
    {"crop": "Cotton", "confidence": 7.2, "kannada": "ಹತ್ತಿ", "icon": "🌿"},
    {"crop": "Groundnut", "confidence": 3.1, "kannada": "ಶೇಂಗಾ", "icon": "🥜"}
  ],
  "soil_analysis": [
    "Optimal pH range",
    "Adequate nitrogen levels",
    "Good phosphorus content",
    "Good potassium levels"
  ]
}
```

---

## 🤖 Machine Learning Model

- **Algorithm**: Random Forest Classifier (sklearn)
- **Training Samples**: 3,600 synthetic samples (300 per crop)
- **Features**: 16 (pH×2, EC×2, OC×2, N×2, P₂O₅×2, K₂O×2, CEC×2, Temperature, Moisture)
- **Classes**: 12 crop types
- **Model Persistence**: Saved as `crop_model.pkl` after first training run

---

## 🎤 Kannada Voice Assistant

- **Speech Input**: Uses Web Speech API (SpeechRecognition) with `kn-IN` locale
- **Text-to-Speech**: Uses Web Speech API (SpeechSynthesis) with Kannada voice
- **Chatbot Backend**: Claude API (claude-sonnet) with Kannada system prompt
- **Fallback**: Keyword-based Kannada responses if API unavailable
- **Browser Support**: Chrome recommended for speech features

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| ML Model | Scikit-learn Random Forest |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Voice Input | Web Speech API |
| Voice Output | Speech Synthesis API |
| Chat AI | Anthropic Claude API |
| Fonts | Google Fonts (Fraunces, DM Sans, Noto Sans Kannada) |

---

## 📝 Notes

- The ML model trains automatically on first run (takes ~10 seconds)
- Model is cached as `backend/models/crop_model.pkl` for subsequent runs
- Kannada voice quality depends on installed TTS voices in the OS/browser
- For best voice support, use Google Chrome on Windows/Linux
- The Kannada assistant uses the Anthropic API — ensure the backend can reach `api.anthropic.com`

---

## 👨‍🎓 Academic Information

**Project Title**: Smart Farming Decision Support System using Machine Learning with Simulated IoT and Regional Voice Assistance

**Technologies Demonstrated**:
- Machine Learning (Random Forest Classification)
- RESTful API Design (FastAPI)
- Natural Language Processing (Regional Language)
- Voice Interface (Speech Recognition & Synthesis)
- Responsive Web Design
