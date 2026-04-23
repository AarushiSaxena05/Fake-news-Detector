from transformers import pipeline

# Load model ONLY once
classifier = pipeline("sentiment-analysis")

def predict(text: str):
    try:
        result = classifier(text)[0]

        label = result["label"]
        score = round(result["score"], 2)

        # Simple mapping (demo purpose)
        if label == "NEGATIVE":
            return {
                "prediction": "Fake News ❌",
                "confidence": score
            }
        else:
            return {
                "prediction": "Real News ✅",
                "confidence": score
            }

    except Exception as e:
        return {"error": str(e)}