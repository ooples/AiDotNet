---
title: "LinkFunctionFactory<T>"
description: "Factory for creating link function instances."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.LinkFunctions`

Factory for creating link function instances.

## For Beginners

This factory creates link functions based on their type.
Choose the link function based on your data:

- Identity: Standard regression (any real values)
- Logit: Binary outcomes (yes/no, 0/1)
- Log: Counts or positive values
- Probit: Binary outcomes (alternative to logit)
- Inverse: Gamma-distributed responses
- CLogLog: Asymmetric probabilities
- Sqrt: Counts with variance stabilization

## How It Works

Use this factory to get the appropriate link function for your GLM.

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(LinkFunctionType)` | Creates a link function of the specified type. |
| `GetAllLinkFunctions` | Gets all available link functions. |
| `GetCanonicalLink(GlmDistributionFamily)` | Gets the canonical link function for a distribution family. |

