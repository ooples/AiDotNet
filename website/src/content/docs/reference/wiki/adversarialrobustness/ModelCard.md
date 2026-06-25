---
title: "ModelCard"
description: "Represents a Model Card for documenting AI model characteristics and performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Documentation`

Represents a Model Card for documenting AI model characteristics and performance.

## For Beginners

A Model Card is like a nutrition label for AI models.
Just as food labels tell you what's in your food and its nutritional value, Model Cards
tell you what the AI model is for, how well it works, what its limitations are, and
any ethical considerations you should know about.

## How It Works

Model Cards provide transparent documentation about machine learning models,
including their intended use, limitations, performance metrics, and ethical considerations.

Based on "Model Cards for Model Reporting" by Mitchell et al. (2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `Caveats` | Gets or sets caveats and additional warnings. |
| `Date` | Gets or sets the date the model was created or last updated. |
| `Developers` | Gets or sets the model developers or organization. |
| `EthicalConsiderations` | Gets or sets ethical considerations and potential biases. |
| `FairnessMetrics` | Gets or sets fairness metrics across different demographic groups. |
| `IntendedUses` | Gets or sets the intended use cases for the model. |
| `Limitations` | Gets or sets known limitations of the model. |
| `ModelName` | Gets or sets the model name and version. |
| `ModelType` | Gets or sets the model type (e.g., "Classification", "Regression", "LLM"). |
| `OutOfScopeUses` | Gets or sets the out-of-scope use cases (what the model should NOT be used for). |
| `PerformanceMetrics` | Gets or sets performance metrics on different datasets. |
| `Recommendations` | Gets or sets recommendations for responsible use. |
| `RobustnessMetrics` | Gets or sets robustness evaluation results. |
| `TrainingData` | Gets or sets the training data description. |
| `Version` | Gets or sets the model version identifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFromEvaluation(String,String,Dictionary<String,Double>,Dictionary<String,Double>)` | Creates a Model Card from evaluation results. |
| `Generate` | Generates a formatted Model Card document. |
| `SaveToFile(String)` | Saves the Model Card to a markdown file. |

