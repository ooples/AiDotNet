---
title: "RegularizationType"
description: "Specifies the type of regularization to apply to a machine learning model."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of regularization to apply to a machine learning model.

## How It Works

**For Beginners:** Regularization is like adding training wheels to your AI model.

When models learn too much from their training data, they might become too specialized
(this is called "overfitting"). Regularization helps prevent this by encouraging the model
to keep things simple.

Think of it like this:

- Without regularization: The model might create very complex rules that work perfectly for

training data but fail on new data.

- With regularization: The model is encouraged to create simpler rules that work well enough

for training data and are more likely to work on new data too.

Different regularization types use different approaches to encourage simplicity.

## Fields

| Field | Summary |
|:-----|:--------|
| `ElasticNet` | A combination of L1 and L2 regularization that balances their properties. |
| `L1` | L1 regularization (also known as Lasso regularization) that encourages sparsity in the model parameters. |
| `L2` | L2 regularization (also known as Ridge regularization) that discourages large parameter values. |
| `None` | No regularization is applied to the model. |

