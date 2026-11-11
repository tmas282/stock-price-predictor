from transformers import pipeline

# The first execution will automatically download the ~440MB model weights file
print("Loading FinBERT model into memory...")
finbert_classifier = pipeline("text-classification", model="ProsusAI/finbert")

if __name__ == "__main__":
    # Test financial text
    text = "The company reported a massive 20% increase in quarterly revenue, beating all market expectations."

    print("Running analysis...")
    result = finbert_classifier(text)

    print(f"\nText: {text}")
    print(f"Sentiment: {result[0]['label']} | Confidence: {result[0]['score']:.4f}")