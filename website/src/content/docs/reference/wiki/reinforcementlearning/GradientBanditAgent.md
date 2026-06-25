---
title: "GradientBanditAgent<T>"
description: "Gradient Bandit agent using softmax action preferences."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Bandits`

Gradient Bandit agent using softmax action preferences.

## For Beginners

Instead of estimating how good each action is (like epsilon-greedy),
the gradient bandit learns preferences for each action using gradient ascent. Actions with
higher preferences are selected more often via softmax probabilities. When an action does
better than average, its preference increases; when worse, it decreases. This approach
naturally handles the exploration-exploitation trade-off through the softmax distribution
without needing an explicit epsilon parameter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientBanditAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

