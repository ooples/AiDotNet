---
title: "MuZeroOptions<T>"
description: "Configuration options for MuZero agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for MuZero agents.

## For Beginners

MuZero is DeepMind's breakthrough that mastered Atari, Go, Chess, and Shogi
without being told the rules. It learns its own "internal model" of the game
and uses tree search to plan ahead.

Key innovations:

- **Learned Model**: No need for game rules, learns environment dynamics
- **MCTS**: Uses Monte Carlo Tree Search for planning
- **Three Networks**: Representation, dynamics, and prediction
- **Planning**: Searches through imagined futures

Think of it like: Learning to play chess by watching games, figuring out
the rules yourself, then planning moves by mentally simulating the game.

Famous for: Superhuman performance across Atari, board games, without rules

## How It Works

MuZero combines tree search (like AlphaZero) with learned models.
It learns dynamics, rewards, and values without knowing environment rules.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MuZeroOptions` | Default constructor required for object-initializer syntax. |
| `MuZeroOptions(MuZeroOptions<>)` | Copy constructor required by the Options golden pattern so Clone() faithfully preserves every property. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Number of discrete actions the agent can choose from. |
| `ObservationSize` | Dimensionality of the environment's observation vector. |
| `Optimizer` | The optimizer used for updating network parameters. |

