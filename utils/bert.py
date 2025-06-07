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
        return tokenizer(batch[column], padding=True, truncation=True, max_length=256)

    train_dataset = Dataset.from_pandas(train_df[[column, label_column]])
    test_dataset = Dataset.from_pandas(test_df[[column, label_column]])
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', label_column])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', label_column])

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Reduce batch size and add gradient accumulation
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 4  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    start_time = time.time()

    best_val_acc = 0.0
    patience = 3
    counter = 0
    max_epochs = 20
    
    # Track metrics
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(max_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Training
        model.train()
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS  # Normalize loss
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            
            # Calculate accuracy
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear GPU memory periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Clear GPU memory periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg training loss = {avg_loss:.4f}, train accuracy = {train_accuracy:.4f}, val accuracy = {val_accuracy:.4f}")

        # Early stopping based on validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.title(f'BERT Model Accuracy Over Time ({column})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(train_accuracies) + 1))  # Show whole numbers for epochs
    history_path = f'data/bert/bert_{column}_history.png'
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    plt.savefig(history_path)
    plt.close()

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
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
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
