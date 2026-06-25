---
title: "OneHotEncodeTransform<T>"
description: "Converts a class index (integer label) to a one-hot encoded vector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms.Numeric`

Converts a class index (integer label) to a one-hot encoded vector.

## For Beginners

One-hot encoding converts a category number into a vector.
For example, with 3 classes: class 0 becomes [1, 0, 0], class 1 becomes [0, 1, 0],
and class 2 becomes [0, 0, 1]. This is required by many neural network loss functions.

## How It Works

Given an integer class index and the total number of classes, produces a vector
of length numClasses where all elements are zero except the element at the class index.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneHotEncodeTransform(Int32)` | Creates a one-hot encoder. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of classes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Int32)` |  |

