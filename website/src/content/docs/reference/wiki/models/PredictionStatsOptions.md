---
title: "PredictionStatsOptions"
description: "Configuration options for prediction statistics generation, which provides statistical analysis and reporting for model predictions including confidence intervals and learning curve analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models`

Configuration options for prediction statistics generation, which provides statistical analysis
and reporting for model predictions including confidence intervals and learning curve analysis.

## For Beginners

Prediction statistics help you understand how reliable your model's predictions are and how your model improves with more data.

Think of prediction statistics like weather forecasting:

- Weather forecasts don't just say "tomorrow will be 75°F"
- They often say "75°F with a 90% chance of being between 72-78°F"
- They also show how forecast accuracy improves with more data points

What these statistics do:

1. Confidence Intervals: Show the range where the true value is likely to fall
- Instead of a single prediction like "house price will be $300,000"
- You get "house price will be $300,000 ± $15,000 with 95% confidence"
- This helps you understand how certain or uncertain each prediction is

2. Learning Curves: Show how your model improves as you give it more training data
- This helps you decide if collecting more data would help your model
- It can reveal if your model has reached its potential or needs more examples

This class lets you configure these statistical measures to better understand your model's performance.

## How It Works

The PredictionStatsOptions class controls how statistical information is calculated and presented
for model predictions. It enables the generation of confidence intervals to quantify prediction
uncertainty and learning curves to track model improvement over increasing training data sizes.
These statistical measures are crucial for understanding model reliability, evaluating prediction
robustness, and determining whether additional training data would improve model performance.
The statistical analysis is particularly valuable for applications in scientific research, 
decision support systems, and critical domains where understanding prediction uncertainty is essential.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceLevel` | Gets or sets the confidence level used for generating prediction confidence intervals. |
| `LearningCurveSteps` | Gets or sets the number of steps used when generating learning curves. |

