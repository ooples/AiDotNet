---
title: "HierarchicalFedLoRA<T>"
description: "Implements HierFedLoRA — Hierarchical LoRA aggregation for edge-cloud federated topologies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Implements HierFedLoRA — Hierarchical LoRA aggregation for edge-cloud federated topologies.

## For Beginners

In a hierarchical FL system, devices first aggregate within
their local "edge" group (e.g., same hospital, same region), then edges aggregate with
the central cloud server. HierFedLoRA applies different LoRA ranks at different levels:
edge clients may use very low ranks (e.g., rank 4) for fast local communication, while the
edge-to-cloud aggregation uses a higher rank to preserve more information.

## How It Works

Topology:

Rank promotion: When edge-level adapters (rank 4) are sent to the cloud, they are
promoted to the higher cloud rank (rank 16) by padding with zeros, then the cloud aggregates
in the higher-rank space and distributes back to edges.

Reference: Hierarchical LoRA Aggregation for Cross-Silo Federated Learning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HierarchicalFedLoRA(Int32,Int32,Int32,Int32,Int32)` | Creates a new HierFedLoRA strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `CompressionRatio` |  |
| `GlobalRank` | Gets the global (cloud) LoRA rank. |
| `LayerDim` | Gets the layer dimension. |
| `LocalRank` | Gets the local (edge) LoRA rank. |
| `NumAdaptedLayers` | Gets the number of adapted layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `AggregateCloud(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>,Boolean)` | Performs cloud-level aggregation of edge adapters. |
| `AggregateEdge(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` | Performs edge-level aggregation of client adapters using weighted averaging at local rank. |
| `AggregateHierarchical(Dictionary<Int32,Dictionary<Int32,Vector<>>>,Dictionary<Int32,Double>,Dictionary<Int32,Double>)` | Full hierarchical aggregation: first aggregate within each edge group, then aggregate across edges at the cloud level. |
| `DemoteToLocalRank(Vector<>)` | Demotes a global-rank adapter back to local rank by truncating the B and A matrices. |
| `ExtractAdapterParameters(Vector<>)` |  |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |
| `PromoteToGlobalRank(Vector<>)` | Promotes a local-rank adapter to the global (cloud) rank by zero-padding the B and A matrices. |

