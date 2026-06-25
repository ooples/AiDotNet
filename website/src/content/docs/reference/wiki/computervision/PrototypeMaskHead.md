---
title: "PrototypeMaskHead<T>"
description: "Prototype-based mask head for YOLO and SOLOv2."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

Prototype-based mask head for YOLO and SOLOv2.

## For Beginners

Instead of predicting masks directly for each instance,
prototype-based methods predict a set of prototype masks and per-instance coefficients.
The final mask is a linear combination of prototypes weighted by coefficients.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypeMaskHead(Int32,Int32)` | Creates a new prototype mask head. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumPrototypes` | Number of mask prototypes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AssembleMask(Tensor<>,Tensor<>)` | Assembles instance mask from prototypes and coefficients. |
| `GeneratePrototypes(Tensor<>)` | Generates prototype masks from feature map. |
| `GetParameterCount` | Gets the total parameter count. |
| `ReadParameters(BinaryReader)` | Reads parameters from binary reader. |
| `WriteParameters(BinaryWriter)` | Writes parameters to binary writer. |

