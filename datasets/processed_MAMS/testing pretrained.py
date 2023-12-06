import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Define the aspect, category, and sentiment labels
aspects = ["aspect1", "aspect2", "aspect3"]
categories = ["category1", "category2", "category3"]
sentiments = ["positive", "neutral", "negative"]

# Sample training data (replace this with your actual dataset)
train_texts = [
    "The product is great.",
    "The service was poor.",
    "I love the ambiance.",
    # Add more examples with different aspects, categories, and sentiments
]
train_aspects = ["aspect1", "aspect2", "aspect3"]
train_categories = ["category1", "category2", "category3"]
train_sentiments = ["positive", "negative", "positive"]

# Create a mapping from sentiment labels to integer values
sentiment_to_label = {sentiment: i for i, sentiment in enumerate(sentiments)}

# Encode the training data using the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

# Convert sentiment labels to integers
train_labels = [sentiment_to_label[sentiment] for sentiment in train_sentiments]
train_encodings["labels"] = train_labels

# Load pre-trained model and modify the classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(sentiments))

# Fine-tune the model
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    output_dir="./aspect_sentiment_model",
    logging_dir="./logs",
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
)

trainer.train()

# Now the model is fine-tuned, and you can use it for inference

def predict_aspect_sentiment(aspect, category, model):
    # Prepare the input text
    input_text = f"{aspect} in {category}."
    inputs = tokenizer(input_text, return_tensors="pt")

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    sentiment_idx = torch.argmax(logits, dim=1).item()
    predicted_sentiment = sentiments[sentiment_idx]

    return predicted_sentiment

# Example usage:
aspect = "product"
category = "electronics"
predicted_sentiment = predict_aspect_sentiment(aspect, category, model)
print(f"The sentiment for {aspect} in {category} is predicted to be: {predicted_sentiment}")
