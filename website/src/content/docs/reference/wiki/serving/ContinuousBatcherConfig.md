---
title: "ContinuousBatcherConfig"
description: "Configuration for the continuous batcher."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Serving.ContinuousBatching`

Configuration for the continuous batcher.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoStart` | Whether to automatically start the batcher when a request is submitted. |
| `EnableSpeculativeDecoding` | Whether to enable speculative decoding. |
| `EosTokenId` | End-of-sequence token ID. |
| `IdleSleepMs` | Milliseconds to sleep when idle. |
| `MaxContextLength` | Maximum number of tokens in context (prompt + generated). |
| `MaxTreeDepth` | Maximum depth of the speculation tree. |
| `SchedulerConfig` | Scheduler configuration. |
| `SpeculationDepth` | Number of tokens to draft ahead when speculative decoding is enabled. |
| `SpeculationPolicy` | Policy for when speculative decoding should run (default: Auto). |
| `SpeculativeMethod` | Speculative decoding method to use (default: Auto). |
| `TreeBranchFactor` | Branching factor for tree speculation (continuations explored per step). |
| `UseTreeSpeculation` | Whether to use tree-based speculation (multiple draft continuations). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForModel(String,Int32)` | Creates config for a specific model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultEosTokenId` | Default end-of-sequence token ID when none is configured. |

