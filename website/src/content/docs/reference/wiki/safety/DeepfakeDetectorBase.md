---
title: "DeepfakeDetectorBase<T>"
description: "Abstract base class for image deepfake detection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Image`

Abstract base class for image deepfake detection modules.

## For Beginners

This base class provides common code for all deepfake detectors.
Each detector type extends this and adds its own way of detecting AI-generated
or manipulated images.

## How It Works

Provides shared infrastructure for deepfake detectors including threshold
configuration and common image analysis utilities. Concrete implementations
provide the actual detection algorithm (frequency, consistency, provenance).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepfakeDetectorBase(Double)` | Initializes the deepfake detector base with a threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDeepfakeScore(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `Threshold` | The detection threshold above which images are flagged as deepfakes. |

