---
title: "AttentiveNAS<T>"
description: "AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling.
Uses an attention-based meta-network to guide the sampling of sub-networks,
focusing search on promising regions of the architecture space.

Reference: "AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling" (CVPR 2021)

## For Beginners

AttentiveNAS improves architecture search by paying
attention to which sub-networks perform well. Instead of sampling random architectures,
it learns to focus on promising designs, like a smart student who concentrates on
the most important study topics rather than studying everything equally.

## Methods

| Method | Summary |
|:-----|:--------|
| `AttentiveSample(Vector<>)` | Samples architecture using attention-based sampling strategy. |
| `ComputeAttentionScores(Vector<>)` | Computes attention scores using the attention module |
| `CreateArchitectureEmbedding(AttentiveNASConfig<>)` | Creates an embedding for an architecture configuration |
| `CreateContextVector` | Creates a context vector from recent architecture performance history |
| `ExtractScores(Vector<>,Int32,Int32)` | Extracts a subset of scores for a specific architecture dimension |
| `GetAttentionWeights` | Gets the attention weights |
| `GetPerformanceMemory` | Gets the performance memory |
| `SampleFromDistribution(List<>)` | Samples from a probability distribution |
| `Search(HardwareConstraints<>,Int32,Int32,Int32)` | Searches for optimal architecture using attentive sampling |
| `Softmax(Vector<>)` | Applies softmax to a vector |
| `UpdateAttention(AttentiveNASConfig<>,,)` | Updates the attention module based on architecture performance. |

