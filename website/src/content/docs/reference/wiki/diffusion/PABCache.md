---
title: "PABCache<T>"
description: "Pyramid Attention Broadcast (PAB) cache for accelerating video diffusion inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Acceleration`

Pyramid Attention Broadcast (PAB) cache for accelerating video diffusion inference.

## For Beginners

PAB (Pyramid Attention Broadcast) Cache speeds up video generation by reusing attention computations across nearby diffusion timesteps. Since adjacent steps produce similar attention patterns, caching avoids redundant computation.

## How It Works

**References:**

- Paper: "Real-Time Video Generation with Pyramid Attention Broadcast" (2024)

PAB exploits the observation that attention outputs in video diffusion models change slowly
across denoising timesteps. Instead of recomputing attention at every step, PAB broadcasts
(reuses) cached attention outputs for a configurable number of steps:

- Spatial attention: changes most slowly, can be broadcast for many steps
- Temporal attention: changes moderately
- Cross-attention: changes most frequently, broadcast for fewer steps

This pyramid of broadcast intervals achieves significant speedup with minimal quality loss.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PABCache(Int32,Int32,Int32)` | Initializes a new PAB cache. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CrossBroadcastInterval` | Gets the cross-attention broadcast interval. |
| `CurrentStep` | Gets the current denoising step. |
| `SpatialBroadcastInterval` | Gets the spatial attention broadcast interval. |
| `TemporalBroadcastInterval` | Gets the temporal attention broadcast interval. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheCross(String,Tensor<>)` | Caches cross-attention output. |
| `CacheSpatial(String,Tensor<>)` | Caches spatial attention output. |
| `CacheTemporal(String,Tensor<>)` | Caches temporal attention output. |
| `GetCachedCross(String)` | Gets cached cross-attention output. |
| `GetCachedSpatial(String)` | Gets cached spatial attention output. |
| `GetCachedTemporal(String)` | Gets cached temporal attention output. |
| `Reset` | Resets the cache for a new generation. |
| `ShouldRecomputeCross(String)` | Checks if cross-attention should be recomputed at the current step. |
| `ShouldRecomputeSpatial(String)` | Checks if spatial attention should be recomputed at the current step. |
| `ShouldRecomputeTemporal(String)` | Checks if temporal attention should be recomputed at the current step. |
| `Step` | Advances to the next denoising step. |

