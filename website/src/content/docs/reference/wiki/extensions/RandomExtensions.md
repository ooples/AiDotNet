---
title: "RandomExtensions"
description: "Provides extension methods for the Random class to generate numbers with specific distributions."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Provides extension methods for the Random class to generate numbers with specific distributions.

## For Beginners

This class adds new capabilities to .NET's built-in Random class.
While the standard Random class can generate uniform random numbers (where every value has an equal chance),
AI and machine learning often need different types of random numbers that follow specific patterns or distributions.

## Methods

| Method | Summary |
|:-----|:--------|
| `NextGaussian(Random)` | Generates a random number from a Gaussian (normal) distribution with mean 0 and standard deviation 1. |
| `NextGaussian(Random,Double,Double)` | Generates a random number from a Gaussian (normal) distribution with specified mean and standard deviation. |

