---
title: "EnsembleJailbreakDetector<T>"
description: "Combines multiple jailbreak detection strategies into a robust ensemble."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Combines multiple jailbreak detection strategies into a robust ensemble.

## For Beginners

Attackers constantly invent new ways to bypass AI safety measures.
By combining multiple detection approaches — pattern matching, semantic analysis, and
encoding detection — this ensemble catches a wider variety of attacks than any single
method could alone.

## How It Works

Runs pattern-based and semantic jailbreak detectors in parallel and aggregates their
findings using weighted voting. A detection from multiple strategies receives a higher
confidence score than a single-strategy detection.

**References:**

- GuardReasoner: Reasoning-based explainable guardrails (2025, arxiv:2501.18492)
- Qwen3Guard: 85.3% accuracy, robust to prompt variation (Alibaba, 2025, arxiv:2510.14276)
- Granite Guardian: 81.0% accuracy with minimal prompt sensitivity (IBM, 2024, arxiv:2412.07724)
- LoRA-Guard: Parameter-efficient customizable guardrails (2024, arxiv:2407.02987)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleJailbreakDetector(Double,Double)` | Initializes a new ensemble jailbreak detector with default sub-detectors. |
| `EnsembleJailbreakDetector(ITextSafetyModule<>[],Double[],Double)` | Initializes a new ensemble jailbreak detector with custom sub-detectors. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

