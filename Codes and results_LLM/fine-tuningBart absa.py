import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

# Load the dataset from the CSV file
df = pd.read_csv("C:/Users/Nimra/OneDrive/ABSA work/Datasets/Reviews Dataset/new_Hotels.csv")

# Load the pre-trained BART tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Define a custom dataset for fine-tuning
class ABSADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["Text"]
        aspect = self.data.iloc[idx]["Aspect"]
        sentiment = self.data.iloc[idx]["Sentiment"]

        # Prepare the input and target text
        input_text = f"<s> {text} </s> <as> {aspect} </s> <st> {sentiment} </s>"
        target_text = f"<s> {aspect} </s> <st> {sentiment} </s>"

        # Tokenize the input and target text
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # Create attention mask manually
        input_attention_mask = torch.ones_like(input_ids)
        target_attention_mask = torch.ones_like(target_ids)

        return {
            "input_ids": input_ids.flatten(),
            "attention_mask": input_attention_mask.flatten(),
            "decoder_input_ids": target_ids.flatten(),
            "decoder_attention_mask": target_attention_mask.flatten(),
        }

# Create the custom dataset
dataset = ABSADataset(df, tokenizer)


# Define the DataLoader for training
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the pre-trained BART model for conditional generation
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Fine-tuning settings
num_epochs = 20
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=decoder_input_ids
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
