---
title: "ResourceMonitor"
description: "Monitors system resources (CPU, memory, GPU) during training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Resources`

Monitors system resources (CPU, memory, GPU) during training.

## How It Works

**For Beginners:** The ResourceMonitor tracks hardware utilization
to help you understand if your training is bottlenecked by resources.

Key metrics:

- CPU Usage: How much processing power is being used
- Memory Usage: RAM consumption and available memory
- GPU Usage: Graphics card utilization (if available)
- GPU Memory: VRAM consumption (if available)

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResourceMonitor(Int32)` | Initializes a new instance of the ResourceMonitor class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsRunning` | Gets whether the monitor is currently running. |
| `MonitorGpu` | Gets or sets whether to attempt GPU monitoring. |
| `NvidiaSmiPath` | Gets or sets the path to nvidia-smi executable. |
| `Thresholds` | Gets or sets the resource warning thresholds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearHistory` | Clears the history. |
| `Dispose` | Disposes the resource monitor. |
| `GetAverage` | Gets average resource usage over the history period. |
| `GetDefaultNvidiaSmiPath` | Gets the default path for nvidia-smi based on the current platform. |
| `GetHistory(Nullable<Int32>)` | Gets the resource history. |
| `GetPeak` | Gets peak resource usage. |
| `GetSnapshot` | Gets the current resource snapshot. |
| `Start(Nullable<TimeSpan>)` | Starts the resource monitor. |
| `Stop` | Stops the resource monitor. |

## Events

| Event | Summary |
|:-----|:--------|
| `ResourceUpdated` | Event raised when resource metrics are updated. |
| `ThresholdExceeded` | Event raised when a resource threshold is exceeded. |

