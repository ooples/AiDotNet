---
title: "FinancialPPOAgentOptions<T>"
description: "Configuration options for the Financial PPO (Proximal Policy Optimization) trading agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Financial PPO (Proximal Policy Optimization) trading agent.

## For Beginners

PPO is a robust policy gradient algorithm that prevents
large policy updates, leading to stable training. These options extend the base
trading agent options with PPO-specific parameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumEpochs` | Number of optimization epochs per batch of data. |
| `NumMiniBatches` | Number of mini-batches for each update. |

