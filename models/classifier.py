import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd

#Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class TicketDataset(Dataset):
    """Custom Dataset class for support ticket classification."""
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        inputs = self.tokenizer.encode_plus(
            row['Ticket Description'],  # Input text
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(row['Ticket Type Label'], dtype=torch.long)  # Target label
        }

#Load datasets
train_data = pd.read_csv("/workspaces/support-ticket-classifier-chatbot/dataset/train_data.csv")
val_data = pd.read_csv("/workspaces/support-ticket-classifier-chatbot/dataset/val_data.csv")

#Prepare datasets for training and validation
train_dataset = TicketDataset(train_data, tokenizer)
val_dataset = TicketDataset(val_data, tokenizer)

#Load the pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(train_data['Ticket Type Label'].unique())  # Number of unique ticket categories
)

#Define training arguments
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",       # Save the model at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,          # Limit the number of checkpoints saved
    load_best_model_at_end=True  # Use the best model based on validation performance
)

#Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

#Train the model
trainer.train()

#Save the trained model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
