from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import os

router = APIRouter()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """ನೀವು ಒಬ್ಬ ಕೃಷಿ ತಜ್ಞ ಸಹಾಯಕ. ನೀವು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಬೇಕು.
ನೀವು ರೈತರಿಗೆ ಈ ವಿಷಯಗಳಲ್ಲಿ ಸಹಾಯ ಮಾಡುತ್ತೀರಿ:
- ಬೆಳೆ ಬೆಳೆಸುವ ವಿಧಾನಗಳು
- ಗೊಬ್ಬರ ಮತ್ತು ರಸಗೊಬ್ಬರ ಬಳಕೆ
- ನೀರಾವರಿ ಪದ್ಧತಿಗಳು
- ಮಣ್ಣಿನ ಸುಧಾರಣೆ
- ಕೀಟ ಮತ್ತು ರೋಗ ನಿಯಂತ್ರಣ
- ಬೆಳೆ ರಕ್ಷಣೆ

ಸರಳ ಮತ್ತು ಸ್ಪಷ್ಟ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ. ರೈತರಿಗೆ ಅರ್ಥವಾಗುವ ಭಾಷೆ ಬಳಸಿ.
You are an agricultural expert assistant. Always respond in Kannada language only."""

class ChatMessage(BaseModel):
    message: str
    history: list = []

@router.post("/chat")
async def chat(data: ChatMessage):
    try:
        messages = []
        for h in data.history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": data.message})
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": SYSTEM_PROMPT,
                    "messages": messages
                }
            )
        
        if response.status_code == 200:
            result = response.json()
            reply = result["content"][0]["text"]
            return {"reply": reply, "success": True}
        else:
            # Fallback Kannada responses
            reply = get_fallback_response(data.message)
            return {"reply": reply, "success": True}
            
    except Exception as e:
        reply = get_fallback_response(data.message)
        return {"reply": reply, "success": True}


def get_fallback_response(message: str) -> str:
    msg_lower = message.lower()
    
    if any(w in message for w in ["ಭತ್ತ", "rice", "ಅಕ್ಕಿ"]):
        return "ಭತ್ತ ಬೆಳೆಯಲು ಮಣ್ಣಿನ pH 5.5-7.0 ಇರಬೇಕು. ಸಾರಜನಕ 80-120 kg/ha, ರಂಜಕ 40-60 kg/ha ಮತ್ತು ಪೊಟ್ಯಾಷ್ 40 kg/ha ನೀಡಿ. ಸಾಕಷ್ಟು ನೀರು ಅಗತ್ಯ."
    elif any(w in message for w in ["ಗೊಬ್ಬರ", "fertilizer", "ರಸಗೊಬ್ಬರ"]):
        return "ಉತ್ತಮ ಇಳುವರಿಗಾಗಿ NPK ಗೊಬ್ಬರ ಬಳಸಿ. ಮಣ್ಣು ಪರೀಕ್ಷೆ ಆಧಾರದ ಮೇಲೆ ಸೂಕ್ತ ಪ್ರಮಾಣದಲ್ಲಿ ನೀಡಿ. ಸಾವಯವ ಗೊಬ್ಬರ (ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ) ಸಹ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಸುಧಾರಿಸುತ್ತದೆ."
    elif any(w in message for w in ["ನೀರು", "irrigation", "ನೀರಾವರಿ"]):
        return "ಹನಿ ನೀರಾವರಿ ಅತ್ಯಂತ ಪರಿಣಾಮಕಾರಿ. ಬೆಳೆಯ ಅಗತ್ಯಕ್ಕೆ ಅನುಗುಣವಾಗಿ ನೀರು ನೀಡಿ. ಅತಿಯಾದ ನೀರು ಬೆಳೆಗೆ ಹಾನಿ ಮಾಡಬಹುದು."
    elif any(w in message for w in ["ಕೀಟ", "pest", "ರೋಗ", "disease"]):
        return "ಕೀಟ ನಿಯಂತ್ರಣಕ್ಕೆ ಜೈವಿಕ ಕೀಟನಾಶಕ ಬಳಸಿ. ಬೆಳೆ ಚಕ್ರ ಅನುಸರಿಸಿ. ರೋಗ ಕಂಡು ಬಂದಾಗ ತಕ್ಷಣ ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
    elif any(w in message for w in ["ಮಣ್ಣು", "soil", "pH"]):
        return "ಮಣ್ಣಿನ pH 6-7 ಉತ್ತಮ. ಆಮ್ಲೀಯ ಮಣ್ಣಿಗೆ ಸುಣ್ಣ ಹಾಕಿ, ಕ್ಷಾರೀಯ ಮಣ್ಣಿಗೆ ಗಂಧಕ ಅಥವಾ ಜಿಪ್ಸಂ ಹಾಕಿ. ಸಾವಯವ ವಸ್ತು ಹೆಚ್ಚಿಸಲು ಹಸಿರು ಗೊಬ್ಬರ ಬೆಳೆಗಳನ್ನು ಬಿತ್ತನೆ ಮಾಡಿ."
    else:
        return f"ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಧನ್ಯವಾದಗಳು. ಕೃಷಿ ಸಂಬಂಧಿತ ಯಾವುದೇ ಮಾಹಿತಿಗಾಗಿ ನಾನು ಸಹಾಯ ಮಾಡಲು ಸಿದ್ಧ. ಬೆಳೆ ಆಯ್ಕೆ, ಗೊಬ್ಬರ, ನೀರಾವರಿ ಅಥವಾ ಕೀಟ ನಿಯಂತ್ರಣದ ಬಗ್ಗೆ ಕೇಳಬಹುದು."
