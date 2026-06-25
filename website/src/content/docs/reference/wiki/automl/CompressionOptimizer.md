---
title: "CompressionOptimizer<T>"
description: "Automatically finds the best compression configuration for a model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Automatically finds the best compression configuration for a model.

## For Beginners

Think of this as an automated assistant that tries different
ways to compress your model and finds the best one for your needs.

Instead of manually trying:

- Different pruning levels (50%, 70%, 90% of weights removed)
- Different quantization settings (8-bit, 5-bit, etc.)
- Different compression techniques (pruning, clustering, Huffman)

The optimizer automatically:

1. Generates compression configurations to try
2. Applies each configuration and measures results
3. Tracks which configurations work best
4. Returns the best compression settings found

Example usage:

## How It Works

CompressionOptimizer uses automated machine learning techniques to find the optimal
compression configuration for a neural network model. It evaluates different compression
techniques and hyperparameters, tracking metrics like compression ratio, accuracy loss,
and inference speed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompressionOptimizer(CompressionOptimizerOptions)` | Initializes a new instance of the CompressionOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestTrial` | Gets the best compression trial found so far. |
| `TrialHistory` | Gets the history of all compression trials. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateCompressor(CompressionType,Dictionary<String,Object>)` | Creates a compression algorithm instance based on the technique and hyperparameters. |
| `GenerateTrials` | Generates a list of compression trials to evaluate. |
| `GetElementSize` | Gets the size in bytes of one element of type T. |
| `GetSummary` | Gets a summary of the optimization results. |
| `Optimize(Vector<>,Func<Vector<>,>)` | Runs the compression optimization process. |

