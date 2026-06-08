# SentimentClassifier
A Python sentiment analysis tool that uses Naive Bayes classification to determine whether text reviews are positive or negative.
 
 
## Features
- **Naive Bayes Classification**: Probabilistic model trained on labeled review data
- **Laplace Smoothing**: Handles unseen words gracefully during classification
- **Evaluation Metrics**: Reports accuracy, precision, recall, and F1 score
- **Stopword Filtering**: Removes common words to focus on sentiment-bearing content
- **Train/Test Split**: Automatically holds out 20% of data for unbiased evaluation
## Installation
Requires Python 3.6+ with the following dependencies:
 
```bash
pip install pandas scikit-learn
```
 
### Quick Start
```bash
# Evaluate classifier performance on a dataset
python classify.py reviews.csv --function evaluate
 
# Display top positive and negative words
python classify.py reviews.csv --function lists
```
 
 
## Usage
### Command-Line Interface
```bash
python classify.py <filepath> [--function <function_name>]
```
 
### Options
- `filepath`: Path to the CSV file containing reviews (required)
- `--function, -f`: Analysis function to run (default: `lists`)
### Input Format
The CSV file must have the following columns:
- `review`: The text of the review
- `sentiment`: The label, either `"positive"` or `"negative"`
### Available Functions
 
**`lists`** (default) — Display the top positive and negative words from the training data
```bash
python classify.py reviews.csv
```
 
**`evaluate`** — Train the classifier and evaluate it on a held-out test set
```bash
python classify.py reviews.csv --function evaluate
```
 
 
## How It Works
### Training Pipeline
1. **Loading**: Reads the CSV and splits into 80% train / 20% test sets
2. **Cleaning**: Lowercases text and strips HTML artifacts
3. **Tokenization**: Extracts words, contractions, abbreviations, and decimals
4. **Filtering**: Removes stopwords to retain meaningful tokens
5. **Counting**: Tallies word frequencies separately for positive and negative reviews
6. **Priors**: Computes class probabilities from the training label distribution
### Classification
Reviews are scored using log-probabilities to avoid floating-point underflow:
 
```
score(class) = log P(class) + Σ log P(word | class)
```
 
Laplace (add-1) smoothing is applied so that words absent from a class do not zero out the score.
 
### Stopwords
The following words are filtered by default:
 
```
i, me, my, we, our, you, your, he, him, she, her, it, its, they, ...
```
 
To modify stopwords, edit the `STOPWORDS` set in `classify.py`.
 
 
## Function Reference
### Core Functions
- `tokenize_words(text)` — Tokenizes text into words, contractions, abbreviations, and decimals
- `clean(text)` — Lowercases text and removes HTML line breaks
- `positive_and_negative_lists(filepath)` — Returns word frequency counters, review counts, and test split
- `train(filepath)` — Trains the model and returns all parameters needed for classification
### Analysis Functions
- `classify(review, ...)` — Returns `"positive"` or `"negative"` for a given review string
- `evaluate(filepath)` — Prints accuracy, precision, recall, and F1 on the held-out test set
- `top_positive_and_negative_words(filepath, top_n=10)` — Prints the most frequent words in each class
## Example Output
 
### `--function lists`
```
==================================================
Top 10 positive words:
--------------------------------------------------
 1. 32779x | film
 2. 29993x | movie
 3. 21560x | one
 ...
==================================================
Top 10 negative words:
--------------------------------------------------
 1. 39516x | movie
 2. 29452x | film
 3. 20966x | one
 ...
==================================================
```
 
### `--function evaluate`
```
==================================================
Total reviews: 10000
--------------------------------------------------
Correct predictions: 8522
--------------------------------------------------
Incorrect predictions: 1478
--------------------------------------------------
Accuracy : 0.8522
--------------------------------------------------
F1 : 0.844421052631579
==================================================
```

 
## Notes
- All text processing is case-insensitive
- HTML artifacts (`<br /><br />`) are stripped automatically during cleaning
- The train/test split uses `random_state=42` for reproducibility
- Short forms: `-f` for `--function`