---
title: "ResourceUsageStats"
description: "Contains statistics about system resource usage during training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Contains statistics about system resource usage during training.

## How It Works

**For Beginners:** This tracks how much of your computer's resources
(CPU, memory, GPU) are being used.

## Properties

| Property | Summary |
|:-----|:--------|
| `CpuUsagePercent` | Gets or sets the CPU usage percentage (0-100). |
| `GpuMemoryUsageMB` | Gets or sets the GPU memory usage in megabytes, if available. |
| `GpuMemoryUsagePercent` | Gets or sets the GPU memory usage percentage (0-100), if available. |
| `GpuUsagePercent` | Gets or sets the GPU usage percentage (0-100), if available. |
| `MemoryUsageMB` | Gets or sets the memory usage in megabytes. |
| `MemoryUsagePercent` | Gets or sets the memory usage percentage (0-100). |
| `Timestamp` | Gets or sets the timestamp when these stats were recorded. |
| `TotalGpuMemoryMB` | Gets or sets the total GPU memory in megabytes, if available. |
| `TotalMemoryMB` | Gets or sets the total available memory in megabytes. |

