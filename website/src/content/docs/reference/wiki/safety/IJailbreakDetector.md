---
title: "IJailbreakDetector<T>"
description: "Interface for jailbreak and prompt injection detection modules."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Text`

Interface for jailbreak and prompt injection detection modules.

## For Beginners

A jailbreak detector catches when someone tries to trick an AI
into ignoring its safety rules. Attackers might use special phrases, encoding tricks,
or gradual escalation to bypass safety measures — this module catches those attempts.

## How It Works

Jailbreak detectors identify attempts to bypass safety measures, override system
instructions, or manipulate AI models through prompt engineering attacks including
direct injection, encoding attacks, multi-turn escalation, and character smuggling.

**References:**

- ShieldGemma: LLM-based safety models (Google DeepMind, 2024, arxiv:2407.21772)
- WildGuard: 13 risk categories, 82.8% accuracy (Allen AI, 2024, arxiv:2406.18495)
- GradSafe: Gradient analysis detecting jailbreaks (2024, arxiv:2402.13494)
- GuardReasoner: Reasoning-based explainable guardrails (2025, arxiv:2501.18492)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetJailbreakScore(String)` | Gets the jailbreak likelihood score for the given text (0.0 = benign, 1.0 = definite jailbreak). |

