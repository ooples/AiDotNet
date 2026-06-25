---
title: "KullbackLeiblerDivergenceFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Kullback-Leibler Divergence to evaluate model performance, particularly for probability distributions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Kullback-Leibler Divergence to evaluate model performance, particularly for probability distributions.

## For Beginners

This calculator helps evaluate how well your model is performing when you're
trying to predict probability distributions (like when your model needs to assign probabilities
to different possible outcomes).

Kullback-Leibler Divergence (often called KL Divergence) measures how different two probability
distributions are from each other. In machine learning, we use it to compare:

- The distribution your model predicted
- The actual distribution from your data

How KL Divergence works:

- It measures the "extra information" needed to represent the actual distribution using your predicted distribution
- It's always non-negative (0 or greater)
- A value of 0 means the distributions are identical (perfect prediction)
- Higher values mean the distributions are more different (worse prediction)

Think of it like this:
Imagine you're trying to guess the weather forecast:

- The actual forecast says: 70% chance of rain, 30% chance of sun
- Your guess is: 60% chance of rain, 40% chance of sun
- KL Divergence measures how "surprised" you would be when you see the actual weather,

given that you were expecting your guessed probabilities

Common applications include:

- Training generative models (like GANs or VAEs)
- Multi-class classification problems
- Natural language processing
- Any task where your model outputs probabilities across multiple categories

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KullbackLeiblerDivergenceFitnessCalculator(DataSetType)` | Initializes a new instance of the KullbackLeiblerDivergenceFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Kullback-Leibler Divergence fitness score for the given dataset. |

