---
title: "WorldModelsOptions<T>"
description: "Configuration options for World Models agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for World Models agents.

## For Beginners

World Models is inspired by how humans learn: we build mental models of the world,
then make decisions based on those models rather than raw sensory input.

Key components:

- **VAE (V)**: Compresses visual observations into compact latent codes
- **MDN-RNN (M)**: Learns temporal dynamics (what happens next)
- **Controller (C)**: Simple linear/neural policy acting in latent space
- **Learning in Dreams**: Agent trains entirely in imagined rollouts

Think of it like: First, learn to compress images (VAE). Then, learn how
compressed images change over time (RNN). Finally, learn to act based on
compressed predictions (controller).

Famous for: Car racing from pixels, learning with limited real environment samples

## How It Works

World Models learns compact spatial and temporal representations using VAE and RNN.
The agent learns entirely within the "dream" of its learned world model.

## Properties

| Property | Summary |
|:-----|:--------|
| `Optimizer` | The optimizer used for updating network parameters. |

