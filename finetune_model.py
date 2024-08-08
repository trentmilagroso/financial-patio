import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load data; Uses FinancialPhraseBank
data_path = r'C:\Users\tmilagroso\Downloads\pythonProj\all-data.csv'  # Update the path as needed
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Check the DataFrame structure
print(df.head())  # Display the first few rows to understand the structure
print(df.columns)  # Check the column names

# Rename columns if necessary (adjust based on actual data)
# If the first column contains sentiments, you can rename it
# For example:
df.columns = ['sentiment', 'text']  # Update this according to your actual data structure

# Map sentiment to numerical labels
label_mapping = {
    'negative': 0,  # Changed from -1 to 0
    'neutral': 1,
    'positive': 2
}

# Update the DataFrame to use the correct column names
if 'sentiment' in df.columns:
    df['label'] = df['sentiment'].map(label_mapping)
else:
    raise KeyError("Column 'sentiment' not found in DataFrame.")

# Check for any unmapped labels
print("Unmapped labels:", df['sentiment'][df['label'].isnull()])

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Load the FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)

# Tokenize the input text
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# Create PyTorch datasets
class FinancialDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FinancialDataset(train_encodings, y_train.tolist())
test_dataset = FinancialDataset(test_encodings, y_test.tolist())

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./finetuned_finbert')
tokenizer.save_pretrained('./finetuned_finbert')
