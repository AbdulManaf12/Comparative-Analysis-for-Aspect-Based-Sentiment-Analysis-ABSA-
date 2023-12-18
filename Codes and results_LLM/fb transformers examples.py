import transformers
from transformers import BartTokenizer, BartForSequenceClassification

# model = transformers.AutoModelForSequenceClassification.from_pretrained("facebook/bart-base")
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name)
def aspect_based_sentiment_analysis(data):
    for review in data:
        # Extract the text and aspects from the review.
        text = review["text"]
        aspects = review["aspects"]

        # Predict the sentiment of each aspect.
        inputs = tokenizer(text=text, max_length=128, return_tensors="pt")
        predictions = model(**inputs)

        # Get the sentiment for each aspect.
        aspect_sentiments = predictions[0].argmax(axis=1)

        # Print the results.
        print(f"Review: {text}")
        print(f"Aspects: {aspects}")
        print(f"Sentiments: {aspect_sentiments}")

if __name__ == "__main__":
    data = [
        {
            "text": "The food was great, but the service was slow.",
            "aspects": ["food", "service"],
            "sentiment": ["positive", "negative"]
        },
        {
            "text": "The hotel was clean and comfortable, but the location was a bit inconvenient.",
            "aspects": ["hotel", "location"],
            "sentiment": ["positive", "negative"]
        }
    ]

    aspect_based_sentiment_analysis(data)
