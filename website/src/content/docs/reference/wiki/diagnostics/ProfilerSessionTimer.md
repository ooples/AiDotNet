---
title: "ProfilerSessionTimer"
description: "A manual profiler timer that must be explicitly stopped."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diagnostics`

A manual profiler timer that must be explicitly stopped.

## Properties

| Property | Summary |
|:-----|:--------|
| `Elapsed` | Gets the elapsed time so far. |
| `Name` | Gets the name of this profiler timer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Stop` | Stops the timer and records the duration. |
| `TryGetThreadAllocatedBytes` | AiDotNet#1355 helper: sample `GC.GetAllocatedBytesForCurrentThread` on platforms that ship it (.NET Core 3.0+ / .NET 5+ / .NET 10) and return -1 on net471 which has no per-thread allocation API. |

