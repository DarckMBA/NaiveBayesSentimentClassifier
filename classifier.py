import re
from collections import Counter
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import math


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
    pos_reviews = 0
    neg_reviews = 0

    for x, y in pairs:
        tokens = tokenize_words(x)
        filtered = [w for w in tokens if re.search(r"\w", w) and w not in STOPWORDS]
        if y == "positive":
            positive_words += Counter(filtered)
            pos_reviews += 1
        if y == "negative":
            negative_words += Counter(filtered)
            neg_reviews += 1

    return positive_words, negative_words, pos_reviews, neg_reviews, x_test, y_test

def train(filepath):
    positive_words, negative_words, pos_reviews, neg_reviews, x_test, y_test = positive_and_negative_lists(filepath)

    # Positive and negative wordcounts
    # positive_words, negative_words

    # Positive and negative total wordcounts
    pos_wordcount = sum(positive_words.values())
    neg_wordcount = sum(negative_words.values())

    pos_and_neg_wordcount = {"positive": pos_wordcount, "negative": neg_wordcount}

    # Vocabulary size
    vocab_size = set(positive_words.keys()) | set(negative_words.keys())
    vocab_size = len(vocab_size)

    # Prior probabilities
    total_reviews = pos_reviews + neg_reviews
    pos_prior = pos_reviews / total_reviews
    neg_prior = neg_reviews / total_reviews

    return positive_words, negative_words, pos_and_neg_wordcount, vocab_size, pos_prior, neg_prior, x_test, y_test

def classify(review, positive_words, negative_words, pos_and_neg_wordcount, vocab_size, pos_prior, neg_prior):
    tokens = tokenize_words(clean(review))

    pos_score = math.log(pos_prior)
    neg_score = math.log(neg_prior)
    for t in tokens:
        pos_score += math.log((positive_words[t] + 1) / (pos_and_neg_wordcount["positive"] + vocab_size))
        neg_score += math.log((negative_words[t] + 1) / (pos_and_neg_wordcount["negative"] + vocab_size))

    winner = ""
    if pos_score > neg_score:
        winner = "positive"
    else:
        winner = "negative"

    return winner


# Function for showing positive and negative words
def top_positive_and_negative_words(filepath, top_n=10):
    positive_words, negative_words, pos_reviews, neg_reviews = positive_and_negative_lists(filepath)

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


# Function for evaluating if a review is positive or negative
def evaluate(filepath):
    positive_words, negative_words, pos_and_neg_wordcount, vocab_size, pos_prior, neg_prior, x_test, y_test = train(filepath)

    test_pairs = zip(x_test, y_test)
    correct = 0
    incorrect = 0
    for x, y in test_pairs:
        prediction = classify(x, positive_words, negative_words, pos_and_neg_wordcount, vocab_size, pos_prior, neg_prior)
        if prediction == y:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect
    accuracy = correct / total

    print("=" * 50)
    print(f"Total reviews: {total}")
    print("-" * 50)
    print(f"Correct predictions: {correct}")
    print("-" * 50)
    print(f"Incorrect predictions: {incorrect}")
    print("-" * 50)
    print(f"Accuracy : {accuracy}")
    print("=" * 50)


# Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentiment in reviews")
    parser.add_argument("filepath", help="Path to the text file to analyze")
    parser.add_argument("--function", "-f", choices=["lists", "evaluate"], default="lists", help="Analysis function to run")
    
    args = parser.parse_args()
    
    if args.function == "lists":
        top_positive_and_negative_words(args.filepath)

    if args.function == "evaluate":
        evaluate(args.filepath)