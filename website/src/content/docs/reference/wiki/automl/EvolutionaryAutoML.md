---
title: "EvolutionaryAutoML<T, TInput, TOutput>"
description: "Built-in AutoML strategy that uses an evolutionary (genetic) approach to propose new trials."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Built-in AutoML strategy that uses an evolutionary (genetic) approach to propose new trials.

## For Beginners

This is like natural selection: keep the best settings, mix them, and make small random changes
to discover even better settings over time.

## How It Works

This strategy treats each trial configuration as an "individual" and iteratively improves by:

- selecting strong prior trials as parents
- combining their settings (crossover)
- randomly tweaking some settings (mutation)

