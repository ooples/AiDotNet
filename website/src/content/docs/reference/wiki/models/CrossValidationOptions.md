---
title: "CrossValidationOptions"
description: "Represents the configuration options for cross-validation in machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Represents the configuration options for cross-validation in machine learning models.

## For Beginners

This class is like a settings panel for cross-validation.

What it does:

- Lets you choose how many parts (folds) to split your data into
- Allows you to pick the type of cross-validation you want to use
- Gives you the option to shuffle your data randomly
- Lets you decide which measurements (metrics) to use when evaluating your model

It's like customizing the rules for a series of tests your model will go through, ensuring you get the most useful 
information about how well your model performs.

## How It Works

This class encapsulates various settings that control how cross-validation is performed. It allows users to customize 
the validation process, including the number of folds, type of validation, data shuffling, and which metrics to compute.
These options provide flexibility in how models are evaluated, enabling users to tailor the validation process to their 
specific needs and dataset characteristics.

## Properties

| Property | Summary |
|:-----|:--------|
| `MetricsToCompute` | Gets or sets the array of metrics to compute during cross-validation. |
| `NumberOfFolds` | Gets or sets the number of folds to use in cross-validation. |
| `RandomSeed` | Gets or sets the random seed for data shuffling. |
| `ShuffleData` | Gets or sets whether to shuffle the data before splitting into folds. |
| `ValidationType` | Gets or sets the type of cross-validation to perform. |

