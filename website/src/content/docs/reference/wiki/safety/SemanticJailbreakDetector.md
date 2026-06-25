---
title: "SemanticJailbreakDetector<T>"
description: "Detects jailbreak attempts using semantic embedding similarity to known attack patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects jailbreak attempts using semantic embedding similarity to known attack patterns.

## For Beginners

Attackers often try to trick AI systems by rephrasing their harmful
requests in creative ways. This module understands the "meaning" of text, not just the
exact words, so it can catch clever rephrasings that simple pattern matching would miss.

## How It Works

Instead of pattern-matching exact phrases, this module computes semantic embeddings of the
input text and compares them against embeddings of known jailbreak attack intents. This
catches rephrased, obfuscated, and novel jailbreak attempts that evade regex-based detection.

**Detection categories:**

1. Role-play injection — "You are now DAN", "Pretend you have no restrictions"
2. Instruction override — "Ignore previous instructions", "Disregard your training"
3. Prompt extraction — Attempts to reveal system prompts or internal instructions
4. Context manipulation — "As an admin", "In developer mode"
5. Gradual escalation — Building trust then escalating harmful requests

**References:**

- GradSafe: Gradient analysis detecting jailbreaks with only 2 examples (2024, arxiv:2402.13494)
- WildGuard: Open moderation covering 13 risk categories (Allen AI, 2024, arxiv:2406.18495)
- ShieldGemma: LLM-based safety models (Google DeepMind, 2024, arxiv:2407.21772)
- Bypassing guardrails: emoji/Unicode smuggling achieving 100% evasion (2025, arxiv:2504.11168)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SemanticJailbreakDetector(Double,Int32)` | Initializes a new semantic jailbreak detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

