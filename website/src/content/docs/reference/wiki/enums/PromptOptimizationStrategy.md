---
title: "PromptOptimizationStrategy"
description: "Represents strategies for optimizing prompts to improve language model performance."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents strategies for optimizing prompts to improve language model performance.

## For Beginners

Prompt optimization strategies automatically improve prompts to get better results.

Think of it like tuning a recipe:

- You start with a basic recipe
- Try different variations (more salt? less sugar? different temperature?)
- Taste test each variation
- Keep the version that tastes best

Prompt optimization does the same thing:

- Start with a basic prompt
- Generate variations
- Test each variation's performance
- Keep the best-performing version

Different strategies use different approaches to search for better prompts.

## Fields

| Field | Summary |
|:-----|:--------|
| `DiscreteSearch` | Discrete optimization that tests variations of prompt components. |
| `Ensemble` | Ensemble multiple prompts and combine their outputs for better performance. |
| `Evolutionary` | Evolutionary optimization using genetic algorithms to evolve better prompts. |
| `GradientBased` | Gradient-based optimization using automatic differentiation (APE - Automatic Prompt Engineering). |
| `MonteCarlo` | Monte Carlo optimization that randomly samples and tests prompt variations. |
| `None` | No optimization - use the prompt as provided. |

