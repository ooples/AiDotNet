---
title: "GuardrailBase<T>"
description: "Abstract base class for guardrail modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Guardrails`

Abstract base class for guardrail modules.

## For Beginners

This base class provides common code for all guardrails.
Each guardrail type extends this and adds its own rules for validating
content entering or leaving the AI system.

## How It Works

Provides shared infrastructure for guardrails including direction configuration
and common content validation utilities. Concrete implementations provide
the actual guardrail logic (input validation, output filtering, topic restriction, custom rules).

## Properties

| Property | Summary |
|:-----|:--------|
| `Direction` |  |

