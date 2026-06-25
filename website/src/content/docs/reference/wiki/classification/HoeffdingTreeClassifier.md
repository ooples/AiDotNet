---
title: "HoeffdingTreeClassifier<T>"
description: "Implements the Hoeffding Tree (Very Fast Decision Tree) for online classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Online`

Implements the Hoeffding Tree (Very Fast Decision Tree) for online classification.

## For Beginners

Hoeffding Tree is a decision tree that can be built incrementally
from streaming data. It uses the Hoeffding bound to determine when enough samples have been
seen to make a confident split decision.

## How It Works

**How it works:**

- Start with a single leaf node
- As samples arrive, update statistics at each leaf
- When enough samples are seen, use Hoeffding bound to decide if the best split is significantly better
- If yes, split the leaf into internal node with two children
- Repeat for all leaves as new samples arrive

**The Hoeffding bound:**
With probability (1 - δ), the true best attribute is within ε of the observed best,
where ε = sqrt(R² * ln(1/δ) / (2n)) and n is the number of samples.

**Advantages:**

- Constant memory per node (bounded statistics)
- Processes each sample only once
- Converges to batch decision tree with enough samples
- Suitable for infinite data streams

**Reference:** Domingos & Hulten, "Mining High-Speed Data Streams" (2000)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HoeffdingTreeClassifier(HoeffdingTreeOptions<>)` | Creates a new Hoeffding Tree classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsWarm` | Gets whether the model has seen at least one sample. |
| `SamplesSeen` | Gets the total number of samples the model has seen. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` | Deserializes the trained Hoeffding tree including all nodes and statistics. |
| `ForceBatchSplits` | Forces greedy splits on all leaves in the tree. |
| `ForceBestSplit(HoeffdingTreeClassifier<>.HoeffdingNode)` | Performs the best split on a leaf node if it has positive information gain, without requiring the Hoeffding bound to be satisfied. |
| `ForceLeafSplits(HoeffdingTreeClassifier<>.HoeffdingNode)` | Recursively attempts splits on all leaf nodes using a greedy criterion (best split with positive information gain), bypassing the Hoeffding bound. |
| `GetOptions` |  |
| `GetParameters` | Returns an empty vector — Hoeffding trees learn structure during training, not flat parameter vectors. |
| `PartialFit(Matrix<>,Vector<>)` | Updates the model with a batch of training samples. |
| `PartialFit(Vector<>,)` | Updates the model with a single training sample. |
| `Predict(Matrix<>)` |  |
| `Serialize` | Serializes the trained Hoeffding tree including all nodes and statistics. |
| `SetParameters(Vector<>)` | No-op — Hoeffding trees learn structure during training, not flat parameter vectors. |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` | Returns a fresh instance — tree structure cannot be reconstructed from flat parameters. |

