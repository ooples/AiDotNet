---
title: "CSPDarknet<T>"
description: "CSP-Darknet backbone network used in YOLO family models (v5, v7, v8)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Backbones`

CSP-Darknet backbone network used in YOLO family models (v5, v7, v8).

## For Beginners

CSP-Darknet is a specialized feature extraction network
designed for real-time object detection. It uses Cross-Stage Partial connections
to reduce computation while maintaining accuracy.

## How It Works

Reference: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CSPDarknet(Double,Double,Int32,IActivationFunction<>)` | Creates a new CSP-Darknet backbone. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Sum across stem + every CSP stage. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeepCopy` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activation` | Activation applied throughout the network. |

