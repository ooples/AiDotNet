---
title: "GradientJailbreakDetector<T>"
description: "Detects gradient-based adversarial jailbreak attacks by analyzing token-level anomalies in the input text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects gradient-based adversarial jailbreak attacks by analyzing token-level anomalies
in the input text.

## For Beginners

Some attackers use mathematical optimization to craft special text
that tricks AI safety filters. This text often looks like random gibberish appended to a
normal prompt. This module detects such gibberish suffixes by checking whether parts of
the text have unusual character patterns.

## How It Works

Gradient-based attacks (GCG, AutoDAN, etc.) optimize adversarial suffixes to trigger
unsafe behavior. These suffixes typically exhibit statistical anomalies detectable without
model gradients: unusual character distributions, high entropy token sequences, abnormal
bigram frequencies, and nonsensical subword patterns. This detector identifies such
adversarial artifacts.

**Detection signals:**

1. Character distribution anomaly — adversarial tokens have non-English character frequencies
2. Bigram entropy — random-looking token sequences have unusually high bigram entropy
3. Subword coherence — GCG tokens are often nonsensical fragments
4. Suffix length anomaly — adversarial suffixes are unusually long appended sequences

**References:**

- GradSafe: Detecting unsafe inputs via safety-critical gradient analysis (2024, arxiv:2402.13494)
- GCG: Universal and transferable adversarial attacks on aligned LMs (2023, arxiv:2307.15043)
- SmoothLLM: Defending LLMs against jailbreaking via randomized smoothing (2024, arxiv:2310.03684)
- Perplexity-based jailbreak detection (Jain et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientJailbreakDetector(Double,Int32)` | Initializes a new gradient-based jailbreak detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBigramEntropyAnomaly(String)` | Measures character bigram entropy. |
| `ComputeCharDistributionAnomaly(String)` | Measures how much the character frequency distribution deviates from English. |
| `ComputeSubwordCoherenceAnomaly(String)` | Measures subword coherence. |
| `EvaluateText(String)` |  |
| `IsNonsensicalToken(String)` | Heuristic check for nonsensical tokens typical of gradient-optimized adversarial text. |

