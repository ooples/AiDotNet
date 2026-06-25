---
title: "TelemetryConfig"
description: "Configuration for telemetry - tracking and monitoring model inference metrics."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for telemetry - tracking and monitoring model inference metrics.

## For Beginners

Telemetry collects data about how your model is performing in production.
Think of it like a fitness tracker for your AI model - it tracks important health metrics so
you know when something goes wrong or when performance degrades.

What gets tracked:

- Latency: How long each inference takes (helps identify slowdowns)
- Throughput: How many inferences per second (measures capacity)
- Errors: When predictions fail (helps identify issues)
- Cache hits/misses: How often cached models are used (optimizes memory)
- Version usage: Which model versions are being used (helps with rollouts)

Why it's important:

- Detect performance degradation before users complain
- Understand usage patterns to optimize resources
- Debug production issues with real data
- Make data-driven decisions about model updates

Privacy Note: Telemetry doesn't collect user data or predictions, only metadata
like timing and version information.

## Properties

| Property | Summary |
|:-----|:--------|
| `CollectErrors` | Alias for TrackErrors for more intuitive access. |
| `CollectLatency` | Alias for TrackLatency for more intuitive access. |
| `CustomTags` | Gets or sets custom tags to include with all telemetry events. |
| `Enabled` | Gets or sets whether telemetry is enabled (default: true). |
| `ExportEndpoint` | Gets or sets the telemetry export endpoint URL (optional). |
| `FlushIntervalSeconds` | Gets or sets the telemetry flush interval in seconds (default: 60). |
| `SamplingRate` | Gets or sets the sampling rate for telemetry (default: 1.0 = 100%). |
| `TrackCacheMetrics` | Gets or sets whether to track cache hit/miss rates (default: true). |
| `TrackDetailedTiming` | Gets or sets whether to track detailed timing breakdowns (default: false). |
| `TrackErrors` | Gets or sets whether to track errors and exceptions (default: true). |
| `TrackLatency` | Gets or sets whether to track inference latency (default: true). |
| `TrackThroughput` | Gets or sets whether to track throughput metrics (default: true). |
| `TrackVersionUsage` | Gets or sets whether to track model version usage (default: true). |

