import re
from collections import Counter
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


# List of words that do not need to be count (from nltk)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'you', 'your', 'yours', 'he', 'him', 'his', 'she',
    'her', 'it', 'its', 'they', 'them', 'what', 'which',
    'who', 'this', 'that', 'am', 'is', 'are', 'was',
    'were', 'be', 'have', 'has', 'had', 'do', 'does',
    'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'can', 'will', 'just', 'should', 'now'
}


# Helper functions
def tokenize_words(text):
    return re.findall(r"(?:[A-Z]\.)+|\d+\.\d+|\w+(?:'\w+)?|[^\w\s]", text)

def clean(text):
    text = text.lower()
    text = text.replace("<br /><br />", "")
    return text

def positive_and_negative_lists(filepath):
    df = pd.read_csv(filepath)
    x = df["review"].apply(clean)
    y = df["sentiment"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    pairs = zip(x_train, y_train)
    positive_words = Counter()
    negative_words = Counter()

    for x, y in pairs:
        tokens = tokenize_words(x)
        filtered = [w for w in tokens if re.search(r"\w", w) and w not in STOPWORDS]
        if y == "positive":
            positive_words += Counter(filtered)
        if y == "negative":
            negative_words += Counter(filtered)

    return positive_words, negative_words


# Function for showing positive and negative words
def top_positive_and_negative_words(filepath, top_n=10):
    positive_words, negative_words = positive_and_negative_lists(filepath)

    print("=" * 50)
    print(f"Top {top_n} positive words:")
    print("-" * 50)

    for rank, (pos_words, pos_freqs) in enumerate(positive_words.most_common(top_n), start=1):
        print(f"{rank:>2}. {pos_freqs:>3}x | {pos_words}")

    print("=" * 50)
    print(f"Top {top_n} negative words:")
    print("-" * 50)

    for rank, (neg_words, neg_freqs) in enumerate(negative_words.most_common(top_n), start=1):
        print(f"{rank:>2}. {neg_freqs:>3}x | {neg_words}")


# Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentiment in reviews")
    parser.add_argument("filepath", help="Path to the text file to analyze")
    parser.add_argument("--function", "-f", choices=["lists"], default="lists", help="Analysis function to run")
    
    args = parser.parse_args()
    
    if args.function == "lists":
        top_positive_and_negative_words(args.filepath)