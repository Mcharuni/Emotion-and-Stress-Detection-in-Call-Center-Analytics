# Emotion and Stress Detection in Call Center Analytics

An end-to-end multimodal analytics pipeline for call centers.

## Features
- Multimodal Fusion (Text sentiment + Audio prosody)
- Batch Analytics on call logs
- FastAPI REST service for real-time scoring

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run batch analytics: `python src/analyzer.py`
3. Start API: `uvicorn src.app:app --reload`
