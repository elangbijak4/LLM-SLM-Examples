import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Step 1: Load and preprocess stock data
# Load stock data (replace 'stock_data.csv' with your dataset)
stock_data = pd.read_csv('stock_data.csv')

# Preprocess stock data (e.g., normalize prices, select relevant features)
# For simplicity, let's assume we have already preprocessed the data and selected relevant features
# You may need to perform more preprocessing depending on your dataset

# Step 2: Tokenization
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize stock data
input_ids = []
labels = []

for index, row in stock_data.iterrows():
    text = row['text']  # Assuming 'text' column contains textual data related to the stock
    label = row['label']  # Assuming 'label' column contains the label (0: decrease, 1: increase)

    # Tokenize text
    encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,  # Add [CLS] and [SEP]
                        max_length = 64,           # Adjust as needed
                        padding='max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

    # Add tokenized text and label to lists
    input_ids.append(encoded_dict['input_ids'])
    labels.append(label)

# Convert lists to tensors
input_ids = torch.cat(input_ids, dim=0)
labels = torch.tensor(labels)

# Step 3: Split dataset into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Step 4: Create DataLoader for training and testing sets
train_dataset = TensorDataset(train_inputs, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_inputs, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 5: Load pre-trained BERT model and define optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 labels: decrease or increase
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: Fine-tune BERT on stock data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3  # Adjust as needed

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        optimizer.zero_grad()

        outputs = model(b_input_ids, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')

# Step 7: Evaluate fine-tuned model on test set
model.eval()
predictions = []
true_labels = []

for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_labels = batch[1]

    with torch.no_grad():
        outputs = model(b_input_ids)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)

    predictions.extend(preds.cpu().numpy())
    true_labels.extend(b_labels.numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy on Test Set: {accuracy * 100:.2f}%')
