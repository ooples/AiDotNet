---
title: "IPromptOptimizer<T>"
description: "Defines the contract for optimizing prompts to improve language model performance."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for optimizing prompts to improve language model performance.

## For Beginners

A prompt optimizer automatically improves your prompts.

Think of it like automatic recipe refinement:

- You start with a basic recipe
- The optimizer tries variations (more salt, less sugar, different temperature)
- It measures which variations taste better
- It keeps refining until it finds the best version

For prompts:

- You provide a basic prompt
- Optimizer generates variations
- Tests each variation's performance
- Returns the best-performing prompt

Example:
Initial prompt: "Classify this review"
After optimization: "Carefully analyze the sentiment and tone of the following product review.
Classify it as positive, negative, or neutral based on the overall customer satisfaction."
Result: 15% accuracy improvement

Benefits:

- Better results without manual trial-and-error
- Discover optimal phrasings you wouldn't think of
- Systematic improvement process
- Measurable performance gains

## How It Works

A prompt optimizer automatically refines prompts to achieve better performance on a specific task.
Optimization strategies include discrete search, gradient-based methods, ensemble approaches,
and evolutionary algorithms.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptimizationHistory` | Gets the optimization history showing performance over iterations. |

