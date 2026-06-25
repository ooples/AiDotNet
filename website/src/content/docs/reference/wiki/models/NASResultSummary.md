---
title: "NASResultSummary"
description: "Represents a redacted summary of Neural Architecture Search (NAS) results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents a redacted summary of Neural Architecture Search (NAS) results.

## For Beginners

After NAS completes, this tells you what architecture was discovered:

- How many layers/nodes were selected
- What operations were chosen at each position
- Any hardware constraints that were considered

## How It Works

This summary captures the discovered architecture and any hardware-aware optimizations
without exposing proprietary implementation details.

## Properties

| Property | Summary |
|:-----|:--------|
| `ArchitectureDescription` | Gets or sets a human-readable description of the discovered architecture. |
| `DiscoveredNodeCount` | Gets or sets the number of nodes/layers in the discovered architecture. |
| `EstimatedFLOPs` | Gets or sets the estimated FLOPs (floating-point operations) of the discovered architecture. |
| `EstimatedParameters` | Gets or sets the estimated parameter count of the discovered architecture. |
| `FinalArchitectureScore` | Gets or sets the final architecture score (validation accuracy or combined metric). |
| `LatencyConstraintMs` | Gets or sets the latency constraint in milliseconds, if specified. |
| `MemoryConstraintMB` | Gets or sets the memory constraint in megabytes, if specified. |
| `OFASubnet` | Gets or sets OnceForAll-specific subnet information when OFA strategy was used. |
| `QuantizationAware` | Gets or sets whether quantization-aware search was enabled. |
| `SearchIterations` | Gets or sets the number of architecture search epochs/iterations completed. |
| `SelectedOperations` | Gets or sets a list of operation types selected at each position in the architecture. |
| `TargetPlatform` | Gets or sets the target hardware platform, if hardware-aware search was used. |

