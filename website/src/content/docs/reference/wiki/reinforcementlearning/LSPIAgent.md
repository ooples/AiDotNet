---
title: "LSPIAgent<T>"
description: "LSPI (Least-Squares Policy Iteration) agent using iterative policy improvement with LSTDQ."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.AdvancedRL`

LSPI (Least-Squares Policy Iteration) agent using iterative policy improvement with LSTDQ.

## For Beginners

LSPI is a batch RL algorithm that makes the most efficient use
of collected data. Instead of learning from one experience at a time, it collects a batch
of experiences and uses linear algebra to find the best policy in one shot. Think of it
like studying all past exam questions at once rather than one at a time. This makes it
very sample-efficient but requires storing all experiences in memory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSPIAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

