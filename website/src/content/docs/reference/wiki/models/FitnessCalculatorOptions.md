---
title: "FitnessCalculatorOptions"
description: "Configuration options for the fitness calculator, which determines how model performance is evaluated."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the fitness calculator, which determines how model performance is evaluated.

## For Beginners

Think of the fitness calculator as a judge that scores how well your AI model
is performing. Just like different sports have different scoring systems (points in basketball, goals in
soccer), AI models can be evaluated using different metrics. Some metrics focus on how close your predictions
are to the actual values, while others might focus on whether your model captures the overall patterns in
the data. These options let you choose which scoring system to use and how to interpret the scores.

## How It Works

The fitness calculator is responsible for computing a score that represents how well a model fits the data
or makes predictions. Different metrics emphasize different aspects of model performance, such as overall
fit, error magnitude, or prediction accuracy.

## Properties

| Property | Summary |
|:-----|:--------|
| `ScoreType` | Gets or sets the type of metric used to calculate the fitness score. |
| `UseMaximumValue` | Gets or sets whether higher values indicate better fitness. |

