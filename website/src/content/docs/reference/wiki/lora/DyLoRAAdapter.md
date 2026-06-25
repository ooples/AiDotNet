---
title: "DyLoRAAdapter<T>"
description: "DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously.

## For Beginners

DyLoRA is like LoRA with a superpower - flexibility!

Standard LoRA problem:

- You choose rank=8 and train
- Later realize rank=4 would work fine (save memory/speed)
- Or need rank=16 for better quality
- Must retrain from scratch with the new rank

DyLoRA solution:

- Train once with multiple ranks (e.g., [2, 4, 8, 16])
- Deploy with ANY of those ranks without retraining
- Switch between ranks at runtime based on device capabilities

How it works:

1. Train with MaxRank (e.g., 16) but randomly use smaller ranks during training
2. Nested dropout ensures each rank works independently
3. After training, pick deployment rank based on needs (2=fastest, 16=best quality)

Use cases:

- Deploy same model to mobile (rank=2) and server (rank=16)
- Dynamic quality scaling based on battery level
- A/B testing different rank/quality trade-offs
- Training once, deploying everywhere

Example: Train with ActiveRanks=[2,4,8], deploy with:

- Rank=2 for mobile devices (98% parameter reduction, good quality)
- Rank=4 for tablets (95% parameter reduction, better quality)
- Rank=8 for desktops (90% parameter reduction, best quality)

## How It Works

DyLoRA extends the standard LoRA approach by training multiple rank configurations simultaneously
using a nested dropout technique. This allows a single trained adapter to be deployed at different
rank levels without retraining, providing flexibility for different hardware constraints or
performance requirements.

The key innovation is nested dropout: during training, for each forward pass, a random rank r
is selected from the active ranks, and only the first r components of matrices A and B are used.
This ensures that smaller ranks can function independently and don't rely on higher-rank components.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DyLoRAAdapter(ILayer<>,Int32,Int32[],Double,Boolean)` | Initializes a new DyLoRA adapter with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActiveRanks` | Gets the array of active ranks used during training. |
| `CurrentDeploymentRank` | Gets or sets the current deployment rank used during inference. |
| `IsTraining` | Gets or sets whether the adapter is in training mode. |
| `MaxRank` | Gets the maximum rank of the DyLoRA adapter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackwardWithRank(Tensor<>,Tensor<>,Int32)` | Performs backward pass through LoRA layer using only the first 'rank' components. |
| `Eval` | Sets the adapter to evaluation mode (uses fixed deployment rank). |
| `Forward(Tensor<>)` | Performs the forward pass with dynamic rank selection. |
| `ForwardWithRank(Tensor<>,Int32)` | Performs forward pass through LoRA layer using only the first 'rank' components. |
| `MaskOutputToRank(Tensor<>,Int32)` | Masks the full LoRA output to only include contributions from the first 'rank' components. |
| `MergeToOriginalLayer` | Merges the DyLoRA adaptation into the base layer using the current deployment rank. |
| `SetDeploymentRank(Int32)` | Sets the deployment rank for inference. |
| `Train` | Sets the adapter to training mode (enables nested dropout). |
| `TrainWithNestedDropout(Tensor<>[],Tensor<>[],Int32,,Func<Tensor<>,Tensor<>,>)` | Trains the adapter with nested dropout across all active ranks. |
| `UpdateParameterGradientsFromLayers` | Updates the parameter gradients vector from the layer gradients. |
| `UpdateParameters()` | Updates parameters for the base layer and the LoRA layer using cached gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activeRanks` | Array of ranks to train simultaneously during nested dropout. |
| `_cachedActiveRank` | Cached active rank from the last forward pass for gradient computation. |
| `_cachedInput` | Cached input from the last forward pass for gradient computation. |
| `_cachedLoRAGradients` | Cached LoRA parameter gradients computed in backward pass. |
| `_currentDeploymentRank` | Current rank to use during inference (forward pass in eval mode). |
| `_isTraining` | Whether the adapter is in training mode (uses nested dropout). |
| `_maxRank` | Maximum rank for the LoRA decomposition. |
| `_random` | Random number generator for nested dropout during training. |

