---
title: "CameraPoseEncoder<T>"
description: "Encodes camera pose (polar, azimuth, radius) into embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Encodes camera pose (polar, azimuth, radius) into embeddings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CameraPoseEncoder(Int32,Nullable<Int32>)` | Initializes a new CameraPoseEncoder. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the number of parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Double,Double,Double)` | Encodes camera pose into an embedding. |
| `GetParameters` | Gets all parameters. |
| `SetParameters(Vector<>)` | Sets all parameters. |

