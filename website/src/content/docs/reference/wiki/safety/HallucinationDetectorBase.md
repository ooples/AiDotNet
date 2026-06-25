---
title: "HallucinationDetectorBase<T>"
description: "Abstract base class for hallucination detection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Text`

Abstract base class for hallucination detection modules.

## For Beginners

This base class provides common code for all hallucination
detectors. Each detector type extends this and adds its own way of checking
whether an AI made something up.

## How It Works

Provides shared infrastructure for hallucination detectors including claim
extraction utilities and common scoring logic. Concrete implementations provide
the actual detection algorithm (reference-based, self-consistency, triplet, entailment).

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAgainstReference(String,String)` |  |
| `ExtractClaims(String)` | Splits text into individual claims (sentences) for per-claim evaluation. |
| `GetHallucinationScore(String)` |  |

