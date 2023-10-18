import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def train_bert_classifier(data, num_labels=3, epochs=3, batch_size=8, learning_rate=1e-5):
    # Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    
    # Tokenize input data
    encodings = tokenizer(list(data['reviews']), truncation=True, padding=True)
    labels = data['sentiment'].tolist()
    
    # Custom dataset class for PyTorch DataLoader
    class SentimentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
    
    # Create dataset and data loader
    dataset = SentimentDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f'Average training loss: {avg_loss:.4f}')
    
    return model

def evaluate_bert_classifier(model, test_data):
    model.eval()
    predictions = []
    true_labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(list(test_data['reviews']), truncation=True, padding=True)
    labels = test_data['sentiment'].tolist()
    
    dataset = SentimentDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy:.2f}')
    return accuracy
