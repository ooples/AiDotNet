---
title: "FedDTCompressor<T>"
description: "Implements FedDT — Decision-tree-based compression for heterogeneous federated architectures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Implements FedDT — Decision-tree-based compression for heterogeneous federated architectures.

## For Beginners

Most gradient compression methods assume all clients have the same
model architecture. FedDT handles heterogeneous architectures by compressing model updates into
decision-tree representations. Each client distills its local model changes into a lightweight
decision tree, sends only the tree structure (much smaller than full gradients), and the server
merges the trees. This enables FL across different model architectures with minimal communication.

## How It Works

Algorithm:

Reference: FedDT: Decision-Tree Compression for Heterogeneous Federated Learning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedDTCompressor(Int32,Int32,Double)` | Creates a new FedDT compressor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTreeDepth` | Gets the maximum tree depth. |
| `MinLeafSize` | Gets the minimum leaf size. |
| `PruningThreshold` | Gets the pruning threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(Dictionary<String,[]>)` | Compresses a parameter update into a decision-tree representation. |
| `Decompress(FedDTCompressor<>.CompressedTree,Dictionary<String,[]>)` | Decompresses a tree representation back into parameter updates. |
| `EstimateCompressionRatio(Int32,FedDTCompressor<>.CompressedTree)` | Estimates the compression ratio achieved by the tree representation. |
| `MergeTrees(Dictionary<Int32,FedDTCompressor<>.CompressedTree>,Dictionary<Int32,Double>,Dictionary<String,[]>)` | Merges multiple compressed trees via weighted averaging of leaf values. |

