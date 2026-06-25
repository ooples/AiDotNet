---
title: "DecisionTransformerAgent<T>"
description: "Decision Transformer agent for offline reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DecisionTransformer`

Decision Transformer agent for offline reinforcement learning.

## For Beginners

Instead of learning "what's the best action", Decision Transformer learns
"what action was taken when the outcome was X". At test time, you specify
the desired outcome, and it generates the action sequence.

Key innovation:

- **Return Conditioning**: Specify target return, get actions that achieve it
- **Sequence Modeling**: Uses transformers like GPT for temporal dependencies
- **No RL Updates**: Just supervised learning on (return, state, action) sequences
- **Offline-First**: Designed for learning from fixed datasets

Think of it as: "Show me examples of successful games, and I'll learn to
generate moves that lead to that level of success."

Famous for: Berkeley/Meta research simplifying RL to sequence modeling

## How It Works

Decision Transformer treats RL as sequence modeling, using transformer architecture
to predict actions conditioned on desired returns-to-go.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |
| `LoadOfflineData(List<List<ValueTuple<Vector<>,Vector<>,>>>)` | Load offline dataset into the trajectory buffer. |
| `SelectActionWithReturn(Vector<>,,Boolean)` | Select action conditioned on desired return-to-go. |
| `Train(Vector<>,Vector<>)` | Supervised (online) training entry from `IFullModel`: trains the transformer to predict `target` (the desired action) from `state`. |

