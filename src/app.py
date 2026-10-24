from fastapi import FastAPI
from pydantic import BaseModel
from src.analyzer import CallCenterAnalyzer

app = FastAPI(title="Call Center Emotion & Stress API", version="1.0")
analyzer = CallCenterAnalyzer()

class CallInput(BaseModel):
    customer_text: str
    speech_tempo_bpm: float
    pitch_mean_hz: float
    energy_rms: float

@app.get("/")
def home():
    return {"message": "Welcome to the Call Center Emotion & Stress Analytics API"}

@app.post("/predict")
def predict_emotion(data: CallInput):
    txt_label, _ = analyzer.analyze_text_sentiment(data.customer_text)
    aud_label, aud_score = analyzer.analyze_audio_stress(
        data.speech_tempo_bpm, data.pitch_mean_hz, data.energy_rms
    )
    risk = "Critical" if "High" in txt_label and "High" in aud_label else "Normal"
    return {
        "text_analysis": txt_label,
        "audio_stress_classification": aud_label,
        "acoustic_stress_score": aud_score,
        "recommended_action": "Escalate to Supervisor" if risk == "Critical" else "Standard Flow"
    }
