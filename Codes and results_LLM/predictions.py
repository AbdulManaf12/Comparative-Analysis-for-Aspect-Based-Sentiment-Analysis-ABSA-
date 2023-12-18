import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the fine-tuned model and tokenizer
model = BartForConditionalGeneration.from_pretrained("fine_tuned_model")
tokenizer = BartTokenizer.from_pretrained("fine_tuned_model")

# Prepare the input text
input_text = "The food was delicious."

# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

# Pass the input text to the model
outputs = model(input_ids=input_ids)

# Extract the aspect and sentiment
aspect = outputs.logits[0, :].argmax().item()
# sentiment = outputs.logits[1, :].argmax().item()

# Convert the aspect and sentiment to strings
aspect = tokenizer.decode(aspect)
# sentiment = tokenizer.decode(sentiment)

print(f"Aspect: {aspect}")