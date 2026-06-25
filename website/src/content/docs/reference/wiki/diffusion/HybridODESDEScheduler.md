---
title: "HybridODESDEScheduler<T>"
description: "Hybrid ODE/SDE scheduler that transitions between deterministic and stochastic sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Hybrid ODE/SDE scheduler that transitions between deterministic and stochastic sampling.

## For Beginners

This scheduler starts deterministically (same seed = same image)
for the main structure, then adds some controlled randomness for fine details.
This gives you both reliable composition and natural-looking textures.

## How It Works

Begins with deterministic ODE sampling for coarse structure, then switches to
stochastic SDE sampling for fine detail generation. The transition point is
configurable, balancing consistency with diversity.

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

