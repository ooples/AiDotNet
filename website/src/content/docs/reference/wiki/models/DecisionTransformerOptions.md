---
title: "DecisionTransformerOptions<T>"
description: "Configuration options for Decision Transformer agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Decision Transformer agents.

## For Beginners

Decision Transformer is a radically different approach to RL. Instead of learning
"what action is best", it learns "what action was taken when the outcome was X".
Then at test time, you tell it "I want outcome X" and it generates actions.

Key innovation:

- **Sequence Modeling**: Uses transformers (like GPT) instead of RL algorithms
- **Return Conditioning**: Specify desired return, get action sequence
- **Offline-Friendly**: Works excellently with fixed datasets
- **No Value Functions**: No Q-networks or critics needed

Think of it like: "Show me examples of successful chess games, and I'll learn
to play moves that lead to success."

Famous for: Berkeley/Meta research showing transformers can replace RL algorithms

## How It Works

Decision Transformer treats RL as sequence modeling, using transformer architecture
to model trajectories conditioned on desired returns.

## Properties

| Property | Summary |
|:-----|:--------|
| `Optimizer` | The optimizer used for updating network parameters. |

