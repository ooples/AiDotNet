---
title: "CompositeGuardrail<T>"
description: "Chains multiple guardrails into a single composite guardrail that runs all child guardrails in sequence and aggregates their findings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

Chains multiple guardrails into a single composite guardrail that runs all child guardrails
in sequence and aggregates their findings.

## For Beginners

Sometimes you want to apply many safety checks at once. Instead of
calling each guardrail separately, CompositeGuardrail bundles them together. Think of it
as a checklist — the text must pass all the checks before it's considered safe.

## How It Works

CompositeGuardrail acts as an orchestrator: it contains a list of child guardrails and
executes them in order. If any child guardrail returns a finding with a blocking action,
evaluation can optionally short-circuit (fail-fast). The direction of the composite
guardrail is determined by the combination of child directions.

**References:**

- Guardrails AI: Validator chaining (2024)
- NeMo Guardrails: Multi-rail pipelines (NVIDIA, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompositeGuardrail(IReadOnlyList<IGuardrail<>>,Boolean,Nullable<GuardrailDirection>)` | Initializes a new composite guardrail. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of child guardrails in this composite. |
| `Direction` |  |
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IGuardrail<>)` | Adds a guardrail to the end of the chain. |
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` |  |

