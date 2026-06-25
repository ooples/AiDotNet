---
title: "RandomSearchAutoML<T, TInput, TOutput>"
description: "AutoML implementation that uses random search over candidate model types and hyperparameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

AutoML implementation that uses random search over candidate model types and hyperparameters.

## For Beginners

This AutoML strategy works like this:

- Pick a model type at random (for example, Random Forest or Logistic Regression).
- Pick a set of settings at random (for example, number of trees).
- Train the model and score it on validation data.
- Repeat and keep the best result.

If you are new to AutoML, random search is a good first choice because it is reliable and easy to reason about.

## How It Works

Random search is a strong baseline for AutoML. It is simple, parallelizable, and often competitive with
more complex search strategies for a given compute budget.

