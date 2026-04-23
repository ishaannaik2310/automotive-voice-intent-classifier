from transformers import pipeline
import pandas as pd

# Load model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Intent labels
intent_labels = [
    "Navigation",
    "Climate Control",
    "Media",
    "Communication",
    "Vehicle Control",
    "General Query"
]

def predict_intent(command):
    result = classifier(command, intent_labels)
    return {
        "Command": command,
        "Predicted Intent": result["labels"][0],
        "Confidence": round(result["scores"][0] * 100, 1)
    }

def run_batch(commands):
    results = [predict_intent(cmd) for cmd in commands]
    df = pd.DataFrame(results)
    
    avg_conf = sum(r["Confidence"] for r in results) / len(results)
    
    return df, avg_conf


# CLI execution
if __name__ == "__main__":
    commands = [
        "Take me to the nearest petrol station",
        "Turn up the air conditioning",
        "Play some hip hop music",
        "Call mom",
        "Turn on the headlights",
        "What is the weather today",
        "Navigate to Mumbai",
        "Set temperature to 22 degrees",
        "Skip this song",
        "Send a message to John",
        "Open the windows",
        "Tell me a fun fact"
    ]

    print("🚗 Automotive Voice Command Intent Classifier")
    print("=" * 50)

    df, avg_conf = run_batch(commands)

    print(df.to_string(index=False))
    print(f"\nAverage Confidence Score: {avg_conf:.1f}%")
