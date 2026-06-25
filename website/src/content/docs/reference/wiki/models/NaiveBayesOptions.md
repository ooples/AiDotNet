---
title: "NaiveBayesOptions<T>"
description: "Configuration options for Naive Bayes classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Naive Bayes classifiers.

## For Beginners

Naive Bayes is one of the simplest and most effective classifiers.

How it works:

1. During training, it learns the probability of each class and the probability of

each feature value given each class

2. During prediction, it uses Bayes' theorem to calculate the probability of each class

given the observed features

3. It returns the class with the highest probability

The "naive" assumption is that features are independent given the class. For example,
in spam detection, the words "free" and "win" might both indicate spam, but the model
assumes they contribute independently to that prediction.

Despite this unrealistic assumption, Naive Bayes often works surprisingly well!

## How It Works

Naive Bayes classifiers are probabilistic classifiers based on Bayes' theorem with the
"naive" assumption of conditional independence between features given the class label.
Despite this simplifying assumption, Naive Bayes often performs well in practice.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the smoothing parameter (alpha) for Laplace/additive smoothing. |
| `ClassPriors` | Gets or sets custom class prior probabilities. |
| `FitPriors` | Gets or sets whether to fit class prior probabilities from the data. |
| `MinVariance` | Gets or sets the minimum variance for Gaussian Naive Bayes. |

