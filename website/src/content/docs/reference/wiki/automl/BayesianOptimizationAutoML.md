---
title: "BayesianOptimizationAutoML<T, TInput, TOutput>"
description: "Built-in AutoML strategy that uses a lightweight Bayesian-style surrogate to guide trial selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Built-in AutoML strategy that uses a lightweight Bayesian-style surrogate to guide trial selection.

## For Beginners

Instead of trying totally random settings every time, this strategy learns from earlier
trials and tries more settings similar to the best ones found so far.

## How It Works

This implementation uses a pragmatic, production-friendly approach:

- Use a bandit policy to allocate trials across candidate model families.
- Use a kernel-weighted surrogate over observed trials to bias sampling toward promising regions.

