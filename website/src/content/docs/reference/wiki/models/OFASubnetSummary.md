---
title: "OFASubnetSummary"
description: "Represents OnceForAll subnet configuration summary."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents OnceForAll subnet configuration summary.

## For Beginners

OnceForAll (OFA) is a NAS technique that trains one large "supernet"
containing many possible architectures. After training, you can extract smaller "subnets"
optimized for specific hardware without retraining. This summary describes the extracted subnet.

## Properties

| Property | Summary |
|:-----|:--------|
| `Depths` | Gets or sets the depth configuration (number of blocks per stage). |
| `EstimatedAccuracy` | Gets or sets the estimated accuracy of the subnet. |
| `EstimatedLatencyMs` | Gets or sets the estimated latency of the subnet on the target platform. |
| `ExpansionRatios` | Gets or sets the expansion ratios per block. |
| `KernelSizes` | Gets or sets the kernel sizes selected per block. |
| `Widths` | Gets or sets the width multipliers per stage. |

