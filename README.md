# automotive-voice-intent-classifier
Transformer-based AI system for classifying in-car voice commands into intents like navigation, media, and climate control using zero-shot learning.
# 🚗 Automotive Voice Intent Classifier

This project simulates an AI-powered in-car voice assistant similar to Cerence AI systems.

It uses a transformer-based zero-shot learning model to classify user commands into predefined intents.

## 🔹 Intents Supported
- Navigation
- Climate Control
- Media
- Communication
- Vehicle Control
- General Query

## 🔹 Example Commands
- "Take me to the nearest petrol station"
- "Play music"
- "Call Mom"
- "Turn on AC"

## 🔹 Tech Stack
- Python
- HuggingFace Transformers (BART MNLI)

## 🔹 How It Works
The system uses zero-shot classification to map natural language commands to intents without training data.

## 🔹 Future Improvements
- Speech-to-text integration
- Real-time assistant
- Edge deployment in vehicles

  ## 🚀 Demo

Run locally:
```bash
streamlit run app_ui.py
