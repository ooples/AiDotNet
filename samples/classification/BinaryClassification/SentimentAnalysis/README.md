# Sentiment Analysis - Movie Reviews

This sample demonstrates binary classification for sentiment analysis using the Multinomial Naive Bayes classifier with TF-IDF-like text features.

## What You'll Learn

- How to use `MultinomialNaiveBayes` for text classification
- How TF-IDF (Term Frequency-Inverse Document Frequency) preprocessing works
- How to evaluate binary classifiers with accuracy, precision, recall, and F1-score
- How to interpret a confusion matrix

## The Problem

Sentiment analysis is the task of determining whether a piece of text expresses a positive or negative sentiment. In this sample:

- **Input**: Movie review text represented as TF-IDF feature vectors
- **Output**: Binary classification (Positive = 1, Negative = 0)

## TF-IDF Preprocessing

TF-IDF converts text into numerical features by:

1. **Term Frequency (TF)**: How often a word appears in a document
2. **Inverse Document Frequency (IDF)**: How rare a word is across all documents
3. **TF-IDF = TF x IDF**: Words that appear frequently in one document but rarely elsewhere get higher scores

For sentiment analysis, words like "amazing" and "terrible" have high discriminative power.

## The Naive Bayes Algorithm

Multinomial Naive Bayes is particularly well-suited for text classification because:

- It handles high-dimensional sparse data efficiently
- It works well with word count/TF-IDF features
- It's fast to train and predict
- It uses Laplace smoothing to handle unseen words

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
=== AiDotNet Sentiment Analysis ===
Binary classification of movie reviews using Naive Bayes

Dataset: 500 movie reviews
Features: 30 words (TF-IDF weighted)

Training set: 400 samples
Test set: 100 samples

Building Multinomial Naive Bayes classifier...
  - Laplace smoothing (alpha=1.0)
  - TF-IDF preprocessing applied to text features

Training classifier...

Classification Metrics:
---------------------------------------------
  Accuracy:  92.00%
  Precision: 91.30%
  Recall:    93.33%
  F1-Score:  92.31%

Confusion Matrix:
---------------------------------------------
              Predicted
              Neg    Pos
  Actual Neg    44      4
  Actual Pos     4     48

Sample Predictions:
---------------------------------------------
  Review 1: Positive (confidence: 87%) [correct]
  Review 2: Negative (confidence: 94%) [correct]
  Review 3: Negative (confidence: 62%) [correct]
  Review 4: Positive (confidence: 98%) [correct]
  Review 5: Negative (confidence: 91%) [correct]

=== Sample Complete ===
```

## Code Highlights

### Creating the Naive Bayes Classifier

```csharp
var nbOptions = new NaiveBayesOptions<double>
{
    Alpha = 1.0,  // Laplace smoothing
    FitPriors = true
};
var classifier = new MultinomialNaiveBayes<double>(nbOptions);
```

### Training and Prediction

```csharp
classifier.Train(xTrain, yTrain);
var predictions = classifier.Predict(xTest);
var probabilities = classifier.PredictProbabilities(xTest);
```

### Calculating Classification Metrics

```csharp
double accuracy = (double)(truePositive + trueNegative) / total;
double precision = (double)truePositive / (truePositive + falsePositive);
double recall = (double)truePositive / (truePositive + falseNegative);
double f1Score = 2 * (precision * recall) / (precision + recall);
```

## Understanding the Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness: (TP + TN) / Total |
| **Precision** | Of predicted positives, how many are correct: TP / (TP + FP) |
| **Recall** | Of actual positives, how many were found: TP / (TP + FN) |
| **F1-Score** | Harmonic mean of precision and recall |

## Confusion Matrix Interpretation

```
              Predicted
              Neg    Pos
  Actual Neg   TN     FP
  Actual Pos   FN     TP
```

- **True Negative (TN)**: Correctly predicted negative
- **False Positive (FP)**: Incorrectly predicted positive (Type I error)
- **False Negative (FN)**: Incorrectly predicted negative (Type II error)
- **True Positive (TP)**: Correctly predicted positive

## Next Steps

- [SpamDetection](../SpamDetection/) - Another binary classification example using SVM
- [IrisClassification](../../MultiClassification/IrisClassification/) - Multi-class classification example
