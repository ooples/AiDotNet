---
title: "BernoulliNaiveBayes<T>"
description: "Bernoulli Naive Bayes classifier for binary/boolean features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.NaiveBayes`

Bernoulli Naive Bayes classifier for binary/boolean features.

## For Beginners

This classifier works best with yes/no, true/false, or present/absent data:

- Does the document contain this word? (yes/no)
- Is this feature present? (0 or 1)
- Does the user have this attribute?

The key difference from Multinomial NB is that Bernoulli NB cares about
absence - it penalizes when a feature that's usually present for a class
is absent in the sample.

Example use cases:

- Document classification with binary word presence (not counts)
- Spam detection with binary features
- Any classification with boolean attributes

## How It Works

Bernoulli Naive Bayes is suitable for classification with binary features
(features that are either 0 or 1, present or absent). It models each feature
as a Bernoulli distribution.

Unlike Multinomial Naive Bayes, Bernoulli NB explicitly models the absence of
features, making it suitable for problems where "not having" a feature is
informative.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BernoulliNaiveBayes(NaiveBayesOptions<>,IRegularization<,Matrix<>,Vector<>>,Double)` | Initializes a new instance of the BernoulliNaiveBayes class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep clone of this model. |
| `ComputeClassParameters(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `ComputeLogLikelihood(Vector<>,Int32)` | Computes the log-likelihood of a sample given a class using Bernoulli distribution. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Serialize` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_binarizeThreshold` | Binarization threshold for converting continuous features to binary. |
| `_logFeatureProbsAbsent` | Log of feature probabilities for absence (P(f=0\|c) = 1 - P(f=1\|c)) for each class. |
| `_logFeatureProbsPresent` | Log of feature probabilities for presence (P(f=1\|c)) for each class. |

