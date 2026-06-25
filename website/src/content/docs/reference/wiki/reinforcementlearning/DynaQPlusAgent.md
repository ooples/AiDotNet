---
title: "DynaQPlusAgent<T>"
description: "Dyna-Q+ agent with exploration bonus for handling changing environments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Planning`

Dyna-Q+ agent with exploration bonus for handling changing environments.

## For Beginners

Dyna-Q+ extends Dyna-Q with an exploration bonus that
encourages revisiting states not seen recently. This is crucial in changing environments
where the optimal strategy may shift over time. The bonus grows with time since last
visit, ensuring the agent periodically re-explores to detect environmental changes.
Think of it as a curious learner who checks old paths to see if anything changed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynaQPlusAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

