---
title: "IQLOptions<T>"
description: "Configuration options for Implicit Q-Learning (IQL) agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Implicit Q-Learning (IQL) agent.

## For Beginners

IQL is designed for offline learning (learning from fixed datasets).
Unlike CQL which adds penalties, IQL uses a clever trick called "expectile regression"
to avoid overestimation.

Key innovation:

- **Expectile Regression**: Focus on upper quantiles of value distribution
- **Implicit Policy Extraction**: No explicit max over actions
- **Simpler than CQL**: Fewer hyperparameters to tune

Think of it like learning the "typical good outcome" rather than the "best possible outcome"
which helps avoid being too optimistic about unseen situations.

Advantages: Simpler, more stable than CQL in many cases

## How It Works

IQL is an offline RL algorithm that avoids explicit policy constraints or
conservative regularization. Instead, it uses expectile regression to extract
a policy from the value function.

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that required properties are set. |

