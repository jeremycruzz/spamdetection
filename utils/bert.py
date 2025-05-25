# bert_model.py
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import torch
from torch import nn, optim

from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset
import matplotlib.pyplot as plt
import os

def run_bert_model(df, column='message', label_column='label', test_size=0.2, random_state=42):
    # Map labels
    df = df.copy()
    df[label_column] = df[label_column].map({'ham': 0, 'spam': 1})

    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch[column], padding=True, truncation=True)

    train_dataset = Dataset.from_pandas(train_df[[column, label_column]])
    test_dataset = Dataset.from_pandas(test_df[[column, label_column]])
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', label_column])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', label_column])

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"))
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    start_time = time.time()

    best_loss = float('inf')
    patience = 2
    counter = 0
    max_epochs = 10

    for epoch in range(max_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg training loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    training_time = time.time() - start_time

    model.eval()
    all_preds = []
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_logits.append(logits.cpu().numpy())
            all_preds.append(np.argmax(logits.cpu().numpy(), axis=1))

    y_pred = np.concatenate(all_preds)
    logits = np.concatenate(all_logits)
    
    y_test = test_df[label_column].to_numpy()
    def softmax_np(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    y_proba = softmax_np(logits)[:, 1]

    print(f'Training Time: {training_time:.4f} seconds')
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix ({column})')
    cf_path = f'data/bert/bert_{column}_cf.png'
    os.makedirs(os.path.dirname(cf_path), exist_ok=True)
    plt.savefig(cf_path)
    print(f"Saved confusion matrix to {cf_path}")
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba, name="BERT")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.legend()
    plt.title(f'ROC AUC Curve ({column})')
    plt.grid(True)
    roc_path = f'data/bert/bert_{column}_roc.png'
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)
    plt.savefig(roc_path)
    print(f"Saved ROC curve to {roc_path}")
    plt.close()

    misclassified_spam = test_df[(y_test == 1) & (y_pred == 0)][column].tolist()
    misclassified_ham = test_df[(y_test == 0) & (y_pred == 1)][column].tolist()

    metrics = {
        'model': 'BERT',
        'variant': f'{column}',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'training_time': training_time,
        'roc_auc': roc_auc_score(y_test, y_proba),
        'misclassified': {
            'spam_as_ham': misclassified_spam,
            'ham_as_spam': misclassified_ham
        }
    }

    return model, tokenizer, metrics
