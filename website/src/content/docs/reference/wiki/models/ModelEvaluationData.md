---
title: "ModelEvaluationData<T, TInput, TOutput>"
description: "Represents a comprehensive collection of evaluation data for a model across training, validation, and test datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a comprehensive collection of evaluation data for a model across training, validation, and test datasets.

## For Beginners

This class stores all the performance information about a model across different datasets.

When evaluating a machine learning model:

- You typically split your data into training, validation, and test sets
- You need to track how well the model performs on each set
- You want to compare performance across these sets to detect issues like overfitting

This class organizes all that information in one place, including:

- Detailed statistics for each dataset (training, validation, test)
- Overall model statistics and metrics

Having this structured collection makes it easier to:

- Evaluate model quality
- Compare different models
- Generate reports and visualizations
- Make informed decisions about model selection and improvement

## How It Works

This class encapsulates all the evaluation data for a machine learning model, including detailed statistics for the 
training, validation, and test datasets, as well as overall model statistics. It provides a complete picture of how 
the model performs across different datasets, which is essential for assessing model quality, diagnosing issues like 
overfitting or underfitting, and making informed decisions about model selection and improvement.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdditionalMetrics` | Gets or sets additional, model-specific scalar metrics keyed by name. |
| `ClusteringMetrics` | Gets or sets the clustering quality metrics, populated when the model is an unsupervised clustering model. |
| `ModelStats` | Gets or sets the overall statistics for the model. |
| `TestSet` | Gets or sets the statistics for the test dataset. |
| `TrainingSet` | Gets or sets the statistics for the training dataset. |
| `ValidationSet` | Gets or sets the statistics for the validation dataset. |

