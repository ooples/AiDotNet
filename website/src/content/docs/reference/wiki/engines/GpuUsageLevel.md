---
title: "GpuUsageLevel"
description: "GPU usage level controlling when to use GPU vs CPU for operations."
section: "API Reference"
---

`Enums` · `AiDotNet.Engines`

GPU usage level controlling when to use GPU vs CPU for operations.

## For Beginners

This controls how aggressively the system uses GPU:

- **Default**: Balanced for typical GPUs (recommended)
- **Conservative**: Only use GPU for very large operations (older/slower GPUs)
- **Aggressive**: Use GPU more often (high-end GPUs like RTX 4090/A100)
- **AlwaysGpu**: Force all operations to GPU (maximize GPU utilization)
- **AlwaysCpu**: Force all operations to CPU (disable GPU entirely)

## Fields

| Field | Summary |
|:-----|:--------|
| `Aggressive` | Aggressive GPU usage - use GPU more often (high-end GPUs). |
| `AlwaysCpu` | Always use CPU for all operations (disable GPU entirely). |
| `AlwaysGpu` | Always use GPU for all operations (maximize GPU utilization). |
| `Conservative` | Conservative GPU usage - only for very large operations (older/slower GPUs). |
| `Default` | Balanced GPU usage - good for most desktop GPUs (default). |

