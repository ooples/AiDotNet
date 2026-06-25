---
title: "CategoricalNaiveBayes<T>"
description: "Categorical Naive Bayes classifier for categorical/discrete features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.NaiveBayes`

Categorical Naive Bayes classifier for categorical/discrete features.

## For Beginners

Use this when your features are categories, like:

- Color: Red, Blue, Green
- Size: Small, Medium, Large
- Weather: Sunny, Rainy, Cloudy

The classifier computes: P(category_value | class) for each feature.

For example, predicting if someone will buy an umbrella:

- P(Rainy | Will Buy) might be high
- P(Sunny | Will Buy) might be low

Features should be encoded as integers (0, 1, 2, ...) representing categories.

Use Categorical NB when:

- Features are truly categorical (not ordinal)
- Features can have more than 2 values (otherwise use Bernoulli)
- Features are not counts (otherwise use Multinomial)

## How It Works

Categorical Naive Bayes handles features that take on discrete categorical values.
Unlike Multinomial NB (which works with counts) or Bernoulli NB (which works with
binary features), Categorical NB handles multi-valued categorical features directly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CategoricalNaiveBayes(NaiveBayesOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the CategoricalNaiveBayes class. |

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
| `_categoryLogProbs` | Category log-probabilities: log P(category\|class). |
| `_numCategories` | Number of categories per feature. |

