---
title: "JailbreakDetectorBase<T>"
description: "Abstract base class for jailbreak and prompt injection detection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Text`

Abstract base class for jailbreak and prompt injection detection modules.

## For Beginners

This base class provides common code for all jailbreak detectors.
Each detector type extends this and adds its own way of catching people trying to
trick the AI into ignoring its safety rules.

## How It Works

Provides shared infrastructure for jailbreak detectors including sensitivity
configuration and common scoring utilities. Concrete implementations provide
the actual detection algorithm (pattern, semantic, gradient, ensemble).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JailbreakDetectorBase(Double)` | Initializes the jailbreak detector base with a sensitivity level. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetJailbreakScore(String)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `Sensitivity` | The detection sensitivity level (0.0 = lenient, 1.0 = strict). |

