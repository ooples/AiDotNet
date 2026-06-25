---
title: "HardwareCostModel<T>"
description: "Models hardware costs for neural architecture search operations using FLOP-based estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

Models hardware costs for neural architecture search operations using FLOP-based estimation.
Supports latency, energy, and memory cost estimation for different hardware platforms.

## Properties

| Property | Summary |
|:-----|:--------|
| `Characteristics` | Gets the platform characteristics used for cost estimation. |
| `Platform` | Gets the target hardware platform. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOperationFlops(String,Int32,Int32,Int32)` | Calculates the number of floating-point operations for a given operation. |
| `CalculateOperationMemory(String,Int32,Int32,Int32)` | Calculates the memory footprint (weights) for a given operation in bytes. |
| `ClearCalibration` | Clears all calibration factors. |
| `EstimateArchitectureCost(Architecture<>,Int32,Int32)` | Estimates the total cost for an entire architecture. |
| `EstimateOperationCost(String,Int32,Int32,Int32)` | Estimates the hardware cost for a given operation using FLOP-based calculation. |
| `GetCalibrationFactor(String)` | Gets the calibration factor for an operation, or 1.0 if not set. |
| `GetCostBreakdown(Architecture<>,Int32,Int32)` | Gets a breakdown of costs per operation in the architecture. |
| `GetTotalFlops(Architecture<>,Int32,Int32)` | Calculates total FLOPs for an architecture. |
| `GetTotalParameters(Architecture<>,Int32)` | Calculates total parameters (weights) for an architecture. |
| `MeetsConstraints(Architecture<>,HardwareConstraints<>,Int32,Int32)` | Checks if an architecture meets the hardware constraints. |
| `SetCalibrationFactor(String,Double)` | Sets a calibration factor for a specific operation based on actual measurements. |

