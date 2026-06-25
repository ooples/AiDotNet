---
title: "CrossModalConsistencyChecker<T>"
description: "Checks consistency between different modalities (text, image, audio) to detect misaligned or manipulated multimodal content."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Multimodal`

Checks consistency between different modalities (text, image, audio) to detect
misaligned or manipulated multimodal content.

## For Beginners

When you have content with both text and images (or audio),
sometimes the text says one thing but the image shows something different or harmful.
This module catches those mismatches — for example, a message that says "cute animals"
but contains a violent image.

## How It Works

Cross-modal attacks exploit the gap between modalities — for example, a benign text
description paired with a harmful image, or a safe-looking image with hidden toxic text.
This module detects such misalignment by comparing safety signals across modalities.

**Detection approach:**

1. Run each modality through its respective safety module independently
2. Compare finding severity across modalities
3. Flag cases where one modality is safe but another is unsafe (potential evasion)
4. Flag cases where text-image semantic similarity is very low (potential mismatch)

**References:**

- OmniSafeBench-MM: 9 risk domains, 50 fine-grained categories (2025)
- MM-SafetyBench: 13 scenarios for multimodal safety (ECCV 2024)
- Cross-modal jailbreak attacks on multimodal LLMs (2024)
- AnyAttack: Transferable adversarial attacks on vision-language models (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossModalConsistencyChecker(Double)` | Initializes a new cross-modal consistency checker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckConsistency(IReadOnlyList<SafetyFinding>,IReadOnlyList<SafetyFinding>,IReadOnlyList<SafetyFinding>)` | Compares safety findings from multiple modalities and returns cross-modal findings. |
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` | Evaluates text for cross-modal consistency indicators. |

