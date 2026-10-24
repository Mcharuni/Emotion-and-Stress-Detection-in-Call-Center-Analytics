import pandas as pd
import numpy as np

class CallCenterAnalyzer:
    def __init__(self, data_path="data/call_logs.csv"):
        self.df = pd.read_csv(data_path)
        
    def analyze_text_sentiment(self, text):
        frustration_keywords = ["unacceptable", "cancel", "immediately", "wrong", "fourth", "three days", "down", "drops"]
        score = sum(1 for word in frustration_keywords if word in text.lower())
        if score >= 2:
            return "High Frustration", 0.85
        elif score == 1:
            return "Moderate Frustration", 0.55
        else:
            return "Neutral/Positive", 0.15

    def analyze_audio_stress(self, tempo, pitch, energy):
        stress_index = (tempo / 160.0 * 0.4) + (pitch / 250.0 * 0.4) + (energy / 0.1 * 0.2)
        if stress_index > 0.7:
            return "High Stress", round(stress_index, 2)
        elif stress_index > 0.45:
            return "Moderate Stress", round(stress_index, 2)
        else:
            return "Low Stress", round(stress_index, 2)

    def process_batch(self):
        results = []
        for _, row in self.df.iterrows():
            txt_label, _ = self.analyze_text_sentiment(row["customer_text"])
            aud_label, aud_score = self.analyze_audio_stress(
                row["speech_tempo_bpm"], row["pitch_mean_hz"], row["energy_rms"]
            )
            risk_score = "Critical" if txt_label == "High Frustration" and aud_label == "High Stress" else "Normal"
            results.append({
                "call_id": row["call_id"], "agent_id": row["agent_id"],
                "text_sentiment": txt_label, "audio_stress_level": aud_label,
                "acoustic_stress_score": aud_score, "overall_call_risk": risk_score
            })
        return pd.DataFrame(results)

if __name__ == "__main__":
    analyzer = CallCenterAnalyzer()
    print(analyzer.process_batch())
