# baseline_model.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay

def run_baseline_model(df, column='cleaned_stop_lemma', label_column='label', test_size=0.2, random_state=42):
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[column])
    y = df[label_column].map({'ham': 0, 'spam': 1})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    test_df = df.iloc[y_test.index]

    # Train model
    import time
    start_time = time.time()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f'Training Time: {training_time:.4f} seconds')

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix ({column})')
    cf_save_path = f'data/baseline/baseline_{column}_cf.png'
    plt.savefig(cf_save_path)
    print(f'Saved confusion matrix to {cf_save_path}')
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba, name='Logistic Regression')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title(f'ROC AUC Curve ({column})')
    plt.grid(True)
    roc_save_path = f'data/baseline/baseline_{column}_roc.png'
    plt.savefig(roc_save_path)
    print(f'Saved ROC curve to {roc_save_path}')
    plt.close()

    misclassified_spam = test_df[(y_test == 1) & (y_pred == 0)][column].tolist()
    misclassified_ham = test_df[(y_test == 0) & (y_pred == 1)][column].tolist()

    metrics = {
        'model': 'Baseline',
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

    return model, vectorizer, metrics