import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import time
import os

def run_lstm_model(df, column='message', label_column='label', test_size=0.2, random_state=42):
    # Map labels
    df = df.copy()
    df[label_column] = df[label_column].map({'ham': 0, 'spam': 1})

    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df[column])
    
    # Convert to sequences
    X_train = tokenizer.texts_to_sequences(train_df[column])
    X_test = tokenizer.texts_to_sequences(test_df[column])
    
    # Pad sequences
    max_length = 256
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')
    
    y_train = train_df[label_column].values
    y_test = test_df[label_column].values

    # Model parameters
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    N_LAYERS = 2
    DROPOUT = 0.5

    # Build model
    model = models.Sequential([
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        layers.Bidirectional(layers.LSTM(HIDDEN_DIM, return_sequences=True)),
        layers.Dropout(DROPOUT),
        layers.Bidirectional(layers.LSTM(HIDDEN_DIM)),
        layers.Dropout(DROPOUT),
        layers.Dense(OUTPUT_DIM, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    training_time = time.time() - start_time

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Training Accuracy')
    plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy Over Time ({column})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(history.history['accuracy']) + 1))  # Show whole numbers for epochs
    history_path = f'data/lstm/lstm_{column}_history.png'
    plt.savefig(history_path)
    plt.close()

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_proba = y_pred[:, 1]

    # Print metrics
    print(f'Training Time: {training_time:.4f} seconds')
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=['Ham', 'Spam']))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix ({column})')
    plt.colorbar()
    plt.xticks([0, 1], ['Ham', 'Spam'])
    plt.yticks([0, 1], ['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    cf_path = f'data/lstm/lstm_{column}_cf.png'
    os.makedirs(os.path.dirname(cf_path), exist_ok=True)
    plt.savefig(cf_path)
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC AUC Curve ({column})')
    plt.legend()
    plt.grid(True)
    roc_path = f'data/lstm/lstm_{column}_roc.png'
    plt.savefig(roc_path)
    plt.close()

    # Get misclassified examples
    misclassified_spam = test_df[(y_test == 1) & (y_pred_classes == 0)][column].tolist()
    misclassified_ham = test_df[(y_test == 0) & (y_pred_classes == 1)][column].tolist()

    metrics = {
        'model': 'LSTM',
        'variant': f'{column}',
        'accuracy': accuracy_score(y_test, y_pred_classes),
        'precision': precision_score(y_test, y_pred_classes),
        'recall': recall_score(y_test, y_pred_classes),
        'f1': f1_score(y_test, y_pred_classes),
        'training_time': training_time,
        'roc_auc': roc_auc_score(y_test, y_proba),
        'misclassified': {
            'spam_as_ham': misclassified_spam,
            'ham_as_spam': misclassified_ham
        }
    }

    return model, tokenizer, metrics
