#!/bin/bash
echo "============================================"
echo " Smart Farming DSS - Linux Setup Script"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Installing..."
    sudo apt update && sudo apt install python3 python3-pip python3-venv -y
fi
echo "[OK] Python3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment."
    exit 1
fi
echo "[OK] Virtual environment created."

# Activate and install
echo ""
echo "Installing dependencies..."
source venv/bin/activate
pip install fastapi uvicorn[standard] python-multipart pydantic scikit-learn numpy pandas httpx edge-tts --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi
echo "[OK] All dependencies installed."

# Start server
echo ""
echo "============================================"
echo " Starting Smart Farming DSS server..."
echo " Open http://localhost:8000 in your browser"
echo " Press Ctrl+C to stop the server"
echo "============================================"
echo ""
cd backend
python main.py
