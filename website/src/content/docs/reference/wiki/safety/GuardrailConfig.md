---
title: "GuardrailConfig"
description: "Configuration for input/output guardrails."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for input/output guardrails.

## For Beginners

Guardrails are safety barriers that check content before and
after model processing. Input guardrails validate requests before they reach the model;
output guardrails validate responses before they reach the user.

## How It Works

**References:**

- ShieldGemma: LLM-based safety models (Google DeepMind, 2024)
- WildGuard: Open moderation covering 13 risk categories (Allen AI, 2024)
- Qwen3Guard: 85.3% accuracy, robust to prompt variation (Alibaba, 2025)

## Properties

| Property | Summary |
|:-----|:--------|
| `InputGuardrails` | Gets or sets whether input guardrails are enabled. |
| `MaxInputLength` | Gets or sets the maximum allowed input length. |
| `OutputGuardrails` | Gets or sets whether output guardrails are enabled. |
| `TopicRestrictions` | Gets or sets restricted topics that should be blocked. |

