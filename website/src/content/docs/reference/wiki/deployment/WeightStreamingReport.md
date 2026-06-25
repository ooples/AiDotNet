---
title: "WeightStreamingReport"
description: "Telemetry summary for a model's weight-streaming activity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Configuration`

Telemetry summary for a model's weight-streaming activity. Issue #1222
task #186. Returned on `AiModelResult.WeightStreamingReport` after
a Build that ran with streaming enabled (whether explicitly via
`ConfigureWeightStreaming` or auto-detected from parameter
count).

## For Beginners

When weight streaming pages model weights
to disk to fit in RAM, this report tells you how much actually
happened during your training/inference run: how many cold disk
reads were needed, how many evictions the LRU pool performed, how
often the prefetcher had the right weights ready vs. caught short.
High eviction counts with low prefetch hits suggest you should bump
the pool capacity; high disk-read counts with stable evictions
suggest you're at steady-state and the working-set fits the budget.

## How It Works

The numbers come from the underlying
`AiDotNet.Tensors.WeightRegistry`'s streaming pool. AiDotNet
wraps them in this DTO so callers can rely on stable property names
across Tensors-side rewrites.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDetected` | True when streaming engaged because `Enabled` was left at `null` and the parameter-count threshold was crossed (vs. |
| `CompressionRatio` | Effective compression ratio achieved by the LZ4-compressed disk-backing store (uncompressed bytes / compressed bytes; higher is better). |
| `CountersUnavailableReason` | Non-null when the streaming-pool counters could not be read at report-build time (e.g. |
| `DiskReadCount` | Number of cold-disk reads the streaming pool performed during the run (cache misses where the weight had to be fetched from the backing store). |
| `EffectiveThresholdParameters` | The threshold the auto-detect compared against. |
| `EvictionCount` | Number of LRU evictions. |
| `ModelParameterCount` | Total parameter count the model declared at the time streaming was configured. |
| `PrefetchHitCount` | Number of prefetches that completed before the requesting layer needed the weights (the win condition — async overlap between disk read and forward compute). |
| `PrefetchIssueCount` | Number of prefetch issues — calls to `WeightRegistry.PrefetchAsync` the forward path made. |
| `PrefetchMissCount` | Number of prefetches that hadn't completed by the time the requesting layer needed the weights — MaterializeScope blocked briefly waiting for the read to finish. |
| `ResidentBytes` | Current resident bytes in the streaming pool — the size of the in-RAM working set at the time the report was captured. |
| `StreamingEnabled` | True when this report was produced by an actual streaming configuration (vs. |

