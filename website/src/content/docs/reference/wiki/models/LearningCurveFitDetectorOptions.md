---
title: "LearningCurveFitDetectorOptions"
description: "Configuration options for the Learning Curve Fit Detector, which analyzes training progress to determine when a model has converged or is unlikely to improve further."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Learning Curve Fit Detector, which analyzes training progress
to determine when a model has converged or is unlikely to improve further.

## For Beginners

When training a machine learning model, it's important to know when
to stop. Training for too long can waste time or cause overfitting (where the model performs
well on training data but poorly on new data), while stopping too early might leave the model
under-trained.

The Learning Curve Fit Detector is like a smart assistant that watches how your model's performance
improves during training. It looks at the pattern of improvement and tries to predict whether
continuing to train will give meaningful benefits or if the model has already learned as much as
it can from the data.

Think of it like watching someone learn a new skill:

- At first, they improve quickly (steep learning curve)
- Over time, the rate of improvement slows down (flattening curve)
- Eventually, they reach a plateau where more practice yields minimal improvement

This class lets you configure how the detector decides when that plateau has been reached,
allowing you to automatically stop training at the right time.

## How It Works

The Learning Curve Fit Detector monitors the training progress of a machine learning model by
analyzing the pattern of error reduction over time. It fits a mathematical curve to the error
values and uses this to predict whether continued training is likely to yield significant
improvements. This can help automatically determine when to stop training.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvergenceThreshold` | Gets or sets the threshold that determines when the model is considered to have converged. |
| `MinDataPoints` | Gets or sets the minimum number of data points (training iterations) required before the detector will attempt to predict convergence. |

