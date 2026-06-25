---
title: "PatternJailbreakDetector<T>"
description: "Pattern-based jailbreak and prompt injection detector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Pattern-based jailbreak and prompt injection detector.

## For Beginners

A "jailbreak" is when someone tries to trick an AI into
ignoring its safety rules. Common tricks include:

- "Ignore your previous instructions and do X"
- "You are now DAN (Do Anything Now)"
- Encoding harmful requests in Base64 or other formats
- Using role-play scenarios to bypass restrictions

This detector catches these patterns using curated rules.

## How It Works

Detects known jailbreak and prompt injection patterns using curated regex rules.
Covers direct injection, role-play attacks, encoding attacks, and instruction override attempts.

**Limitations:** Pattern-based detection catches known attack formats but cannot
detect novel or semantically sophisticated jailbreaks. For production deployments,
combine with embedding-based or gradient-based detectors.

**References:**

- Bypassing LLM Guardrails: emoji/Unicode smuggling achieving 100% evasion

(2025, arxiv:2504.11168) — motivates need for multi-strategy detection

- GradSafe: Gradient analysis detecting jailbreaks with only 2 examples

(2024, arxiv:2402.13494) — future enhancement direction

- WildGuard: Open moderation covering 13 risk categories, 82.8% accuracy

(Allen AI, 2024, arxiv:2406.18495)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PatternJailbreakDetector(Double)` | Initializes a new instance of the pattern-based jailbreak detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

