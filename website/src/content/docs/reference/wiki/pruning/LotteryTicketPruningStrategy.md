---
title: "LotteryTicketPruningStrategy<T>"
description: "Implements the Lottery Ticket Hypothesis (Frankle & Carbin, 2019)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Pruning`

Implements the Lottery Ticket Hypothesis (Frankle & Carbin, 2019).

## For Beginners

This strategy is based on a fascinating discovery in neural networks!

The Lottery Ticket Hypothesis says:
"Inside every large neural network, there's a smaller 'winning lottery ticket' network
that could have achieved the same performance if trained from the start."

The analogy:
Imagine you're building a team:

- You start with 100 people (full network)
- After working together, you realize only 20 people did most of the work
- If you had started with just those 20 people from day one, you'd achieve the same results!

How it works:

1. Train the network to completion
2. Find which weights became large (important)
3. Create a mask keeping only those weights
4. Reset the KEPT weights to their original random initialization
5. Retrain with the mask - you'll match the original accuracy!

This is different from regular pruning because:

- Regular pruning: Train → Prune → Fine-tune
- Lottery ticket: Train → Prune → Reset to init → Retrain from scratch

Why this matters:

- Shows that the structure (which connections) matters more than learned values
- Enables training sparse networks from scratch
- Challenges assumptions about why neural networks work

Example workflow:

1. Initialize network with random weights W₀
2. Train to get final weights W_final
3. Create mask M based on |W_final| (keep largest 30%)
4. Reset: W = W₀ ⊙ M (original weights, masked)
5. Retrain this sparse network - it matches full network performance!

## How It Works

The Lottery Ticket Hypothesis states that dense neural networks contain sparse subnetworks
(winning tickets) that, when trained in isolation from initialization, can match the performance
of the original network. This strategy finds these winning tickets through iterative pruning
and resetting weights to their initial values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LotteryTicketPruningStrategy(Int32)` | Creates a new lottery ticket pruning strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsStructured` | Gets whether this is structured pruning (false for lottery ticket). |
| `Name` | Gets the name of this pruning strategy. |
| `RequiresGradients` | Gets whether this strategy requires gradients (false for lottery ticket). |
| `SupportedPatterns` | Gets supported sparsity patterns (unstructured and N:M patterns). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyPruning(Matrix<>,IPruningMask<>)` | Applies the pruning mask to weights in-place. |
| `ApplyPruning(Tensor<>,IPruningMask<>)` | Applies pruning mask to tensor weights in-place. |
| `ApplyPruning(Vector<>,IPruningMask<>)` | Applies pruning mask to vector weights in-place. |
| `BuildKeepIndicesFromMasked(Vector<>)` | Builds a boolean array indicating which indices currently have non-zero values. |
| `ComputeImportanceScores(Matrix<>,Matrix<>)` | Computes importance scores using magnitude-based scoring. |
| `ComputeImportanceScores(Tensor<>,Tensor<>)` | Computes importance scores using magnitude-based scoring for tensors. |
| `ComputeImportanceScores(Vector<>,Vector<>)` | Computes importance scores using magnitude-based scoring for vectors. |
| `CountNonZero(Matrix<>)` | Counts the number of non-zero elements in a matrix. |
| `CountNonZero(Vector<>)` | Counts the number of non-zero elements in a vector. |
| `Create2to4Mask(Tensor<>)` | Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible). |
| `CreateMask(Matrix<>,Double)` | Creates a pruning mask using iterative magnitude pruning. |
| `CreateMask(Tensor<>,Double)` | Creates a pruning mask using iterative magnitude pruning for tensors. |
| `CreateMask(Vector<>,Double)` | Creates a pruning mask using iterative magnitude pruning for vectors. |
| `CreateNtoMMask(Tensor<>,Int32,Int32)` | Creates an N:M structured sparsity mask. |
| `GetInitialWeights(String)` | Gets the stored initial weights for a layer. |
| `GetSortedNonZeroScores(Vector<>)` | Gets non-zero scores sorted by value in ascending order. |
| `PruneLowestScores(Boolean[],List<ValueTuple<Int32,Double>>,Int32)` | Marks the lowest-scoring indices as pruned by setting their keep flags to false. |
| `ResetToInitialWeights(String,Matrix<>,IPruningMask<>)` | Resets pruned weights to their initial values (key step in lottery ticket). |
| `StoreInitialWeights(String,Matrix<>)` | Stores initial weights before training (critical for lottery ticket). |
| `ToSparseFormat(Tensor<>,SparseFormat)` | Converts pruned weights to sparse format for efficient storage. |
| `ToSparseFormat(Tensor<>,SparseFormat,Int32,Int32)` | Converts pruned weights to N:M structured sparse format for efficient storage. |

