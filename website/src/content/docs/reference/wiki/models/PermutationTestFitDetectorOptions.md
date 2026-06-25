---
title: "PermutationTestFitDetectorOptions"
description: "Configuration options for the permutation test fit detector, which helps identify overfitting, underfitting, and high variance in machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the permutation test fit detector, which helps identify overfitting,
underfitting, and high variance in machine learning models.

## For Beginners

The permutation test fit detector is a tool to check if your AI model is learning properly.

Imagine you're teaching someone to identify birds:

- You want them to learn actual bird characteristics (feathers, beak shape, etc.)
- Not just memorize the specific pictures you showed them

What this detector does:

- It takes your data and randomly shuffles it many times
- It checks how your model performs on the shuffled data
- It compares this to performance on the real, unshuffled data
- If your model only works well on the original arrangement, it's learning real patterns
- If it works equally well on random arrangements, it might be "cheating" or guessing

This helps detect three common problems:

1. Overfitting: Your model memorized examples instead of learning general rules

(like knowing what a robin looks like only if it's in the exact same pose as your training image)

2. Underfitting: Your model is too simple to capture the patterns in your data

(like only using bird size to identify species, ignoring color, shape, etc.)

3. High Variance: Your model gives inconsistent results with small data changes

(like completely changing its bird identification if the lighting is slightly different)

This class lets you configure different aspects of this testing process.

## How It Works

The permutation test is a statistical method used to evaluate model performance by comparing
the actual model performance against performance on randomly shuffled (permuted) data. This approach
helps determine if the model is truly learning patterns in the data or merely capitalizing on random
correlations. By running multiple permutation tests, we can establish statistical confidence in our
model's performance and detect common issues like overfitting, underfitting, and high variance.
The permutation test is particularly valuable when working with small datasets or when evaluating
complex models where traditional validation methods might be insufficient.

## Properties

| Property | Summary |
|:-----|:--------|
| `HighVarianceThreshold` | Gets or sets the threshold for detecting high variance. |
| `NumberOfPermutations` | Gets or sets the number of random permutations to perform during the test. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting. |
| `SignificanceLevel` | Gets or sets the statistical significance level for the permutation test. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting. |

