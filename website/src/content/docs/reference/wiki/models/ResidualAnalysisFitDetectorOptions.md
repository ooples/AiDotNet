---
title: "ResidualAnalysisFitDetectorOptions"
description: "Configuration options for the Residual Analysis Fit Detector, which evaluates model fit quality by analyzing prediction residuals against various statistical thresholds."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models`

Configuration options for the Residual Analysis Fit Detector, which evaluates model fit quality
by analyzing prediction residuals against various statistical thresholds.

## For Beginners

This class helps you decide if your prediction model is doing a good job.

When a model makes predictions, it's rarely perfect. The differences between what your model 
predicted and the actual values are called "residuals." Analyzing these residuals helps determine 
if your model is working well.

Think of it like this:

- You have a weather app that predicts temperatures
- Some days it predicts 75°F when the actual temperature is 73°F (residual of -2°F)
- Other days it predicts 68°F when the actual temperature is 72°F (residual of +4°F)
- By analyzing all these differences, you can tell if your model is reliable

This class lets you set thresholds for different statistical measures:

- How close the average residual should be to zero
- How consistent the residuals should be (not too scattered)
- How small the percentage errors should be
- How much of the data variation your model explains

If your model's residuals stay within these thresholds, it passes the "fit test" and is 
considered reliable for making predictions.

## How It Works

Residual analysis is a critical technique in regression modeling that examines the differences between 
observed values and predicted values (residuals) to assess model fit quality. This class provides 
configuration options for threshold values used to determine whether a model's residuals indicate 
a good fit. The detector evaluates several statistical measures including the mean of residuals, 
standard deviation of residuals, Mean Absolute Percentage Error (MAPE), and the coefficient of 
determination (R²). By adjusting these thresholds, users can control how strictly the detector 
evaluates model fit according to their specific requirements and domain knowledge.

## Properties

| Property | Summary |
|:-----|:--------|
| `MapeThreshold` | Gets or sets the threshold for the Mean Absolute Percentage Error (MAPE). |
| `MeanThreshold` | Gets or sets the threshold for the mean (average) of residuals. |
| `StdThreshold` | Gets or sets the threshold for the standard deviation of residuals. |

