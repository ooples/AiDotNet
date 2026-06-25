---
title: "IncrementalState<T>"
description: "Represents the internal state of a time series transformer for incremental processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents the internal state of a time series transformer for incremental processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferFilled` | Whether the buffer has been fully filled at least once. |
| `BufferPosition` | The current position (index) in the circular buffer. |
| `ExtendedState` | Additional state information specific to the transformer type. |
| `PointsProcessed` | The number of data points that have been processed. |
| `RollingBuffer` | The rolling buffer of recent values for each input feature. |

