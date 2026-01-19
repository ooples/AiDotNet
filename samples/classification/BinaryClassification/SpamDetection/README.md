# Spam Detection - Email Classification

This sample demonstrates binary classification for email spam detection using a Support Vector Machine (SVM) classifier.

## What You'll Learn

- How to use `SupportVectorClassifier` for binary classification
- How to preprocess features with `StandardScaler`
- Understanding SVM hyperparameters (C, kernel, gamma)
- Interpreting confusion matrix for spam detection
- Balancing precision vs recall for spam filtering

## The Problem

Spam detection is a classic binary classification problem:

- **Input**: Email features (word counts, special characters, sender info, etc.)
- **Output**: Binary classification (Spam = 1, Ham/Legitimate = 0)

### Why Spam Detection is Challenging

1. **Class Imbalance**: Usually more legitimate emails than spam
2. **False Positive Cost**: Blocking legitimate emails is worse than missing some spam
3. **Evolving Patterns**: Spammers constantly change tactics
4. **Feature Engineering**: Choosing the right features is critical

## Features Used

| Feature | Description |
|---------|-------------|
| word_count | Number of words in email |
| char_count | Total character count |
| uppercase_ratio | Ratio of uppercase letters |
| exclamation_count | Number of ! characters |
| link_count | Number of URLs/hyperlinks |
| contains_free | Binary: contains word "free" |
| contains_winner | Binary: contains word "winner" |
| sender_in_contacts | Binary: sender is known |
| reply_to_mismatch | Binary: reply-to differs from sender |
| html_content_ratio | Ratio of HTML to plain text |

## The SVM Algorithm

Support Vector Machines find the optimal hyperplane that separates classes:

- **Support Vectors**: The data points closest to the decision boundary
- **Margin**: Distance between support vectors and the hyperplane
- **Kernel Trick**: Maps data to higher dimensions for non-linear separation

### Key Hyperparameters

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **C** | Regularization | Higher = stricter boundary, risk of overfitting |
| **kernel** | Transformation type | RBF for non-linear, Linear for simple |
| **gamma** | RBF kernel width | Higher = more complex boundary |

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
=== AiDotNet Spam Detection ===
Binary classification of emails using Support Vector Machine

Dataset: 500 emails
Features: 15 extracted features

Class distribution:
  - Spam: 200 (40.0%)
  - Ham (legitimate): 300 (60.0%)

Training set: 400 samples
Test set: 100 samples

Preprocessing: Standardizing features...

Building Support Vector Classifier...
  - RBF kernel
  - C = 1.0 (regularization)
  - gamma = auto

Training SVM classifier (this may take a moment)...

Classification Metrics:
--------------------------------------------------
  Accuracy:    91.00%
  Precision:   88.64% (spam detection rate)
  Recall:      92.50% (spam catch rate)
  Specificity: 90.00% (legitimate email protection)
  F1-Score:    90.53%

Confusion Matrix:
--------------------------------------------------
                        Predicted
                     Ham      Spam
  Actual Ham           54         6
  Actual Spam           3        37

Metrics Interpretation:
--------------------------------------------------
  - 6 legitimate emails incorrectly marked as spam
  - 3 spam emails that got through
  - 37 spam emails correctly blocked
  - 54 legitimate emails correctly delivered

Sample Predictions:
--------------------------------------------------
  Email  1: Predicted=Ham  Actual=Ham  (conf: 12%) [correct]
  Email  2: Predicted=SPAM Actual=SPAM (conf: 89%) [correct]
  ...

Top Spam Indicators (Feature Analysis):
--------------------------------------------------
  Contains 'FREE'        ||||||||||||||||| 85%
  Exclamation count      |||||||||||||| 72%
  Uppercase ratio        ||||||||||||| 68%
  Contains '$$$'         ||||||||||||| 65%
  Link count             |||||||||||| 61%

=== Sample Complete ===
```

## Code Highlights

### Creating the SVM Classifier

```csharp
var svmOptions = new SVMOptions<double>
{
    C = 1.0,
    Kernel = KernelType.RBF,
    Gamma = 0.1,
    Tolerance = 1e-3,
    MaxIterations = 1000,
    RandomState = 42
};
var classifier = new SupportVectorClassifier<double>(svmOptions);
```

### Feature Standardization

```csharp
var scaler = new StandardScaler<double>();
xTrain = scaler.FitTransform(xTrain);
xTest = scaler.Transform(xTest);
```

### Training and Prediction

```csharp
classifier.Train(xTrain, yTrain);
var predictions = classifier.Predict(xTest);
var probabilities = classifier.PredictProbabilities(xTest);
```

## Understanding Spam Detection Metrics

### Precision vs Recall Trade-off

- **High Precision**: Few legitimate emails marked as spam, but some spam gets through
- **High Recall**: Most spam is caught, but some legitimate emails may be blocked

For spam detection, **precision is often more important** because:
- Blocking a legitimate email (false positive) is very costly
- Missing some spam (false negative) is annoying but acceptable

### Confusion Matrix for Spam

```
                  Predicted
               Ham    Spam
Actual Ham      TN      FP  <- False Positive: Legitimate email marked as spam!
Actual Spam     FN      TP  <- True Positive: Spam correctly caught
```

## Tuning for Better Results

### If Too Many False Positives (blocking legitimate emails):
- Increase decision threshold
- Use higher C value
- Add more features to distinguish legitimate patterns

### If Too Many False Negatives (spam getting through):
- Lower decision threshold
- Increase gamma for RBF kernel
- Add more spam indicator features

## Next Steps

- [SentimentAnalysis](../SentimentAnalysis/) - Binary classification with Naive Bayes
- [IrisClassification](../../MultiClassification/IrisClassification/) - Multi-class classification example
