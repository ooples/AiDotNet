---
title: "TextToxicityDetectorBase<T>"
description: "Abstract base class for toxicity detection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Text`

Abstract base class for toxicity detection modules.

## For Beginners

This base class provides shared code for all toxicity detectors.
Each detector type (rule-based, ML-based, etc.) extends this class and adds its own
detection method while reusing common threshold and scoring logic.

## How It Works

Provides shared infrastructure for toxicity detectors including the toxicity
threshold configuration and common scoring utilities. Concrete implementations
provide the actual detection algorithm (rule-based, embedding, classifier, ensemble).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextToxicityDetectorBase(Double)` | Initializes the toxicity detector base with a threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetToxicityScore(String)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `Threshold` | The toxicity threshold above which content is flagged. |

