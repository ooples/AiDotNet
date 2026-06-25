---
title: "AdversarialRobustnessEvaluator<T>"
description: "Evaluates text inputs for adversarial perturbations designed to evade safety filters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Adversarial`

Evaluates text inputs for adversarial perturbations designed to evade safety filters.

## For Beginners

People sometimes try to sneak harmful content past AI safety filters
by using visual tricks — like replacing "a" with "а" (Cyrillic "a" that looks identical),
inserting invisible characters between letters, or using numbers/symbols to spell words.
This module catches those tricks.

## How It Works

Detects a variety of adversarial text manipulation techniques: homoglyph substitution
(replacing characters with visually similar Unicode), leetspeak encoding, zero-width
character insertion, text direction override attacks, and invisible character padding.
These techniques attempt to bypass keyword-based safety filters.

**References:**

- TextFool: Fool NLP models with adversarial text (2024)
- Universal adversarial triggers for attacking NLP (Wallace et al., EMNLP 2019)
- Homoglyph attacks on content moderation (2024)
- Unicode security considerations (UTS #39, Unicode Consortium)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialRobustnessEvaluator(Double)` | Initializes a new adversarial robustness evaluator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` |  |

