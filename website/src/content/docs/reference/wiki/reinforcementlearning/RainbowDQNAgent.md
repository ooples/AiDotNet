---
title: "RainbowDQNAgent<T>"
description: "Rainbow DQN agent combining six extensions to DQN."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Rainbow`

Rainbow DQN agent combining six extensions to DQN.

## For Beginners

Rainbow takes the best ideas from six different DQN improvements and combines them.
It's currently the strongest DQN variant, achieving state-of-the-art performance.

Six components:

1. **Double Q-learning**: Reduces overestimation
2. **Dueling Architecture**: Separates value and advantage
3. **Prioritized Replay**: Samples important experiences more
4. **Multi-step Returns**: Better credit assignment
5. **Distributional RL (C51)**: Learns distribution of returns
6. **Noisy Networks**: Parameter noise for exploration

Famous for: DeepMind's combination achieving human-level Atari performance

## How It Works

Rainbow combines: Double Q-learning, Dueling networks, Prioritized replay,
Multi-step learning, Distributional RL (C51), and Noisy networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RainbowDQNAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |
| `Predict(Vector<>)` | IFullModel.Predict surfaces the raw Q-value vector (one element per discrete action) rather than the one-hot committed action. |
| `SoftmaxAtomSlice(Vector<>,Int32,Int32)` | Numerically-stable softmax over `logits`[`offset`..`offset`+`count`). |
| `Train(Vector<>,Vector<>)` | Supervised Train(state, target) does one direct regression step on the online Q-network: minimise ‖Q(state, ·) − target‖² so callers that provide an explicit Q-value target — offline pretraining from logged data, distillation, BC warm start… |

