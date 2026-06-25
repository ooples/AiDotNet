---
title: "ClassifierOptions<T>"
description: "Configuration options for classification models, which are machine learning methods used to predict categorical outcomes (discrete classes) rather than continuous values."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for classification models, which are machine learning methods used to
predict categorical outcomes (discrete classes) rather than continuous values.

## For Beginners

Classification is about predicting categories, not numbers.

Think of examples like:

- Is this email spam or not? (Binary classification)
- What type of animal is in this picture? (Multi-class classification)
- What topics does this article cover? (Multi-label classification)
- How satisfied is this customer? (Ordinal classification)

This class lets you configure how the classification model is set up and trained.

## How It Works

Classification is a supervised learning technique where the goal is to predict which category
or categories a sample belongs to. This class provides base configuration options for all
classification models, with specific classifiers potentially extending these options.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassWeights` | Gets or sets custom class weights for each class. |
| `TaskType` | Gets or sets the type of classification task. |
| `UseClassWeights` | Gets or sets whether to use class weights to handle imbalanced datasets. |

