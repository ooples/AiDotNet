---
title: "IInputGradientComputable<T>"
description: "Interface for models that support computing gradients with respect to input data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for models that support computing gradients with respect to input data.

## For Beginners

When training a model, we compute gradients to adjust the model's internal parameters (weights).
This interface instead computes how sensitive the output is to changes in the input data.

For example, if you have an image classifier:

- Parameter gradients tell us how to adjust weights to improve accuracy
- Input gradients tell us which pixels, if changed, would most affect the prediction

This is essential for adversarial robustness testing - we can find the smallest image change
that fools the classifier.

## How It Works

This interface enables models to compute how their output changes with respect to input modifications.
Unlike `IGradientComputable` which computes gradients for model parameters,
this interface computes gradients for the input itself.

**Use Cases:**

