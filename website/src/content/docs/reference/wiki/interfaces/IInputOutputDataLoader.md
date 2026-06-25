---
title: "IInputOutputDataLoader<T, TInput, TOutput>"
description: "Interface for data loaders that provide standard input-output (X, Y) data for supervised learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for data loaders that provide standard input-output (X, Y) data for supervised learning.

## For Beginners

Most machine learning tasks fall into this pattern:

**Example: House Price Prediction**

- X (inputs): Square footage, number of bedrooms, location, age
- Y (outputs): The actual house price

**Example: Email Spam Detection**

- X (inputs): Email text features (word counts, sender info, etc.)
- Y (outputs): Label (spam=1, not spam=0)

The data loader loads this data from files, databases, or other sources
and provides it in the format your model needs for training.

## How It Works

This interface is for standard supervised learning scenarios where you have:

- Input features (X): The data used to make predictions
- Output labels (Y): The correct answers the model should learn to predict

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` | Gets the number of features per sample. |
| `Features` | Gets all input features as a single data structure. |
| `Labels` | Gets all output labels as a single data structure. |
| `OutputDimension` | Gets the number of output dimensions (1 for regression/binary classification, N for multi-class with N classes). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Double,Double,Nullable<Int32>)` | Creates a train/validation/test split of the data. |

