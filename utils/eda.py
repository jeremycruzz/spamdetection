import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2
import seaborn as sns
from collections import Counter
import pandas as pd


def plot_message_lengths(df, column="message", label_column="label", save_path=None, title = None):
    import numpy as np
    spam_lengths = df[df[label_column] == 'spam'][column].str.len()
    ham_lengths = df[df[label_column] == 'ham'][column].str.len()

    # Define uniform bins for both plots
    bucket_size = 10
    max_length = max(spam_lengths.max(), ham_lengths.max())
    bins = np.arange(0, max_length + bucket_size, bucket_size)

    max_length = max(spam_lengths.max(), ham_lengths.max())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

    sns.histplot(spam_lengths, bins=bins, kde=False, color='tomato', ax=axes[0])
    
    axes[0].set_title(f"Spam Message Lengths ({title})")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, None)

    sns.histplot(ham_lengths, bins=bins, kde=False, color='skyblue', ax=axes[1])
    
    axes[1].set_title(f"Ham Message Lengths ({title})")
    axes[1].set_xlabel("Length")
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(0, None)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_top_words_by_class(df, column, label_column='label', label_value='spam', top_n=20, save_path=None, title=None):
    tokens = df[df[label_column] == label_value][column].str.split().sum()
    counter = Counter(tokens)
    top_words = counter.most_common(top_n)
    words, counts = zip(*top_words)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.xticks(rotation=45)
    if title:
        plt.title(title)
    else:
        plt.title(f"Top {top_n} Words in {label_value.title()} Messages ({column})")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def plot_venn_words(df, column="message", label_column="label", save_path=None, title=None):
    spam_tokens = set(df[df[label_column] == "spam"][column].str.split().sum())
    ham_tokens = set(df[df[label_column] == "ham"][column].str.split().sum())

    plt.figure(figsize=(8, 6))
    venn2([spam_tokens, ham_tokens], set_labels=('Spam', 'Ham'))
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()