---
title: "ComplementNaiveBayes<T>"
description: "Complement Naive Bayes classifier designed for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.NaiveBayes`

Complement Naive Bayes classifier designed for imbalanced datasets.

## For Beginners

Think of it this way: instead of asking "how likely is this word in spam?",
CNB asks "how unlikely is this word in NOT-spam?"

This helps when:

- One class has many more examples than others
- Features are not uniformly distributed across classes
- Standard Multinomial NB is biased toward the majority class

Example: In text classification with 95% non-spam and 5% spam,
standard NB might always predict non-spam. CNB corrects this.

CNB is particularly effective for:

- Text classification with imbalanced classes
- Sentiment analysis
- Topic categorization

## How It Works

Complement Naive Bayes (CNB) addresses some of the drawbacks of the standard
Multinomial Naive Bayes, particularly on imbalanced datasets. Instead of
computing P(feature|class), it computes P(feature|NOT class).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComplementNaiveBayes(NaiveBayesOptions<>,IRegularization<,Matrix<>,Vector<>>,Boolean)` | Initializes a new instance of the ComplementNaiveBayes class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeClassParameters(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `ComputeLogLikelihood(Vector<>,Int32)` | Computes the log-likelihood for a sample given a class. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Serialize` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_complementLogProbs` | Complement feature log-probabilities: log P(feature\|NOT class). |
| `_normalize` | Whether to normalize feature weights. |

