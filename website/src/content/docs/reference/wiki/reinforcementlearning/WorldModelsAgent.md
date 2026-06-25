---
title: "WorldModelsAgent<T>"
description: "World Models agent learning compact representations with VAE and RNN."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.WorldModels`

World Models agent learning compact representations with VAE and RNN.

## For Beginners

World Models is inspired by how humans learn: we build mental models
of the world, then make decisions based on those models rather than
raw sensory input.

Three components (V-M-C):

- **V (VAE)**: Compresses visual observations into compact codes
- **M (MDN-RNN)**: Learns temporal dynamics (what happens next)
- **C (Controller)**: Simple policy acting in latent space
- **Learning in Dreams**: Trains entirely in imagined rollouts

Process: First compress images (VAE), then learn how compressed
images change over time (RNN), finally learn to act based on
compressed predictions (controller).

Famous for: Car racing from pixels with limited environment samples

## How It Works

World Models learns compact spatial and temporal representations.
Agent trains entirely in the "dream" of its learned world model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WorldModelsAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AssignDeterministicSeeds(List<ILayer<>>,Int32)` | Assigns a deterministic, reproducible `RandomSeed` to each layer (base seed + index) so lazy weight initialization is identical across agent instances built from the same options — required for Clone() to reproduce the original's policy. |
| `GetOptions` |  |

