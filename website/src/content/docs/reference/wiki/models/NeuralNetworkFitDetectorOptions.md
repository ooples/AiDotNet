---
title: "NeuralNetworkFitDetectorOptions"
description: "Configuration options for the Neural Network Fit Detector, which evaluates the quality of a neural network's fit to data by analyzing performance metrics and detecting issues like underfitting and overfitting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Neural Network Fit Detector, which evaluates the quality of a neural network's
fit to data by analyzing performance metrics and detecting issues like underfitting and overfitting.

## For Beginners

The Neural Network Fit Detector helps you understand if your AI model is learning properly.

When training an AI model, three common problems can occur:

- Underfitting: The model is too simple and performs poorly on all data
- Overfitting: The model has "memorized" the training data instead of learning general patterns
- Just Right: The model has learned general patterns that work well on new data

Imagine you're teaching someone to play chess:

- Underfitting is like they only learned how pieces move, but no strategies
- Overfitting is like they memorized specific games but can't adapt to new situations
- Just Right is when they learned general principles they can apply to any game

This class lets you set thresholds that determine:

- What level of error is acceptable for a "good" model
- When a model is considered "moderately good"
- When a model is performing poorly
- When a model shows signs of overfitting (performing much better on training data than on new data)

These thresholds help automatically classify models during development and training,
so you can quickly identify and fix issues in your neural network.

## How It Works

The Neural Network Fit Detector provides automated detection and classification of model fit quality
based on configurable thresholds. It analyzes the discrepancy between training and validation performance
to identify issues such as underfitting (poor performance on both training and validation data) and
overfitting (good performance on training data but poor performance on validation data). This tool
helps data scientists and machine learning engineers quickly assess model quality and make informed
decisions about architecture adjustments, regularization techniques, or data augmentation strategies.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the maximum error threshold for classifying a model's fit as "good". |
| `ModerateFitThreshold` | Gets or sets the maximum error threshold for classifying a model's fit as "moderate". |
| `OverfittingThreshold` | Gets or sets the threshold for detecting overfitting based on the difference between training and validation performance. |
| `PoorFitThreshold` | Gets or sets the maximum error threshold for classifying a model's fit as "poor" rather than "very poor". |

