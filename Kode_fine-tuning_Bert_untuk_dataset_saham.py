import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

# Baca dataset
df = pd.read_csv("stock_data_with_news.csv")

# Inisialisasi tokenizer dan model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Split data menjadi training dan validation
train_size = int(0.8 * len(df))
train_data, valid_data = df[:train_size], df[train_size:]

class StockDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['News']
        inputs = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        labels = torch.tensor(self.data.iloc[idx]['Close'])
        return inputs, labels

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3

# Datasets and Dataloaders
train_dataset = StockDataset(train_data, tokenizer, MAX_LEN)
valid_dataset = StockDataset(valid_data, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs = {key: val.squeeze(1) for key, val in inputs.items()}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    
    # Validation loop
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_dataloader:
            inputs = {key: val.squeeze(1) for key, val in inputs.items()}
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits.squeeze(), labels.float())
            valid_loss += loss.item()
    valid_loss /= len(valid_dataloader)
    
    print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')
