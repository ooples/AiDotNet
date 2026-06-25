---
title: "MultinomialNaiveBayes<T>"
description: "Multinomial Naive Bayes classifier for discrete count data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.NaiveBayes`

Multinomial Naive Bayes classifier for discrete count data.

## For Beginners

This classifier works best with count data - things you can count, like:

- Number of times each word appears in a document (text classification)
- Number of each type of event
- Frequency of features in categorical data

It's the go-to algorithm for spam detection and sentiment analysis!

During training, it learns how often each feature occurs in each class.
During prediction, it calculates which class is most likely given the
observed feature counts.

Example use cases:

- Spam detection (word counts in emails)
- Topic classification of documents
- Sentiment analysis (positive/negative word counts)

## How It Works

Multinomial Naive Bayes is suitable for classification with discrete features
representing counts or frequencies (e.g., word counts in text classification).
It models each feature as a multinomial distribution.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultinomialNaiveBayes(NaiveBayesOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the MultinomialNaiveBayes class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep clone of this model. |
| `ComputeClassParameters(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `ComputeLogLikelihood(Vector<>,Int32)` | Computes the log-likelihood of a sample given a class using multinomial distribution. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Serialize` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_logFeatureProbs` | Log of feature probabilities for each class. |

