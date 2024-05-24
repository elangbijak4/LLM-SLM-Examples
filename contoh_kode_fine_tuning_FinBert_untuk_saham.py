import pandas as pd
import torch
from transformers import FinBERTForSequenceClassification, FinBERTTokenizer

# Load the FinBERT model and tokenizer
model = FinBERTForSequenceClassification.from_pretrained('finbert')
tokenizer = FinBERTTokenizer.from_pretrained('finbert')

# Load your dataset (replace with your own data)
df = pd.read_csv('stock_data.csv')

# Preprocess the data
texts = df['text']  # assume 'text' column contains news articles or financial reports
labels = df['label']  # assume 'label' column contains stock price labels (e.g., 0 or 1)

# Tokenize the text data
input_ids = []
attention_masks = []
for text in texts:
    encoding = tokenizer.encode_plus(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoding['input_ids'].flatten())
    attention_masks.append(encoding['attention_mask'].flatten())

input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

# Create a custom dataset class for our data
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

# Create a dataset and data loader
dataset = StockDataset(input_ids, attention_masks, labels)
batch_size = 32
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):  # train for 5 epochs
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

# Evaluate the model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.scores, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(dataset)
print(f'Test Loss: {test_loss / len(data_loader)}')
print(f'Test Accuracy: {accuracy:.4f}')
