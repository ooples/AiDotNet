---
title: "MuZeroAgent<T>"
description: "MuZero agent combining tree search with learned models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.MuZero`

MuZero agent combining tree search with learned models.

## For Beginners

MuZero is DeepMind's breakthrough that achieved superhuman performance in
Atari, Go, Chess, and Shogi without being told the rules. It learns its own
"internal model" of the game and uses tree search to plan ahead.

Three key networks:

- **Representation**: Observation -> hidden state
- **Dynamics**: (hidden state, action) -> (next hidden state, reward)
- **Prediction**: hidden state -> (policy, value)

Plus tree search (MCTS) for planning using the learned model.

Think of it as: Learning chess by watching games, figuring out the rules
yourself, then planning moves by mentally simulating the game.

Famous for: Superhuman Atari/board games without knowing rules

## How It Works

MuZero combines tree search (like AlphaZero) with learned dynamics.
It masters games without knowing the rules, learning its own internal model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MuZeroAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a parameter-identical copy of this agent. |
| `GetOptions` |  |
| `Predict(Vector<>)` | IFullModel.Predict surfaces the raw prediction-network output (policy logits + value) for the input observation rather than the one-hot committed action. |

