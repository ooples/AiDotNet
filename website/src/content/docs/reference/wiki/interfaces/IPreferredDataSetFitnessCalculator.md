---
title: "IPreferredDataSetFitnessCalculator<T, TInput, TOutput>"
description: "Optional extension interface for fitness calculators that have a preferred dataset type."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Optional extension interface for fitness calculators that have a preferred dataset type.
Optimizers can probe for this interface to skip computing stats on datasets the calculator
doesn't actually use, avoiding the dominant cost on the optimizer hot path.

## How It Works

This is intentionally split from `IFitnessCalculator` to
avoid breaking external implementations that predate the optimizer perf work. Calculators
that don't implement this interface fall back to the optimizer's default
(`Validation`) — the historical default, so behavior is preserved
for older callers.

All in-repo calculators (`FitnessCalculatorBase` and friends) implement this interface,
so any user that subclasses our base class gets the optimization for free.

## Properties

| Property | Summary |
|:-----|:--------|
| `PreferredDataSetType` | The dataset type this calculator uses for fitness scoring. |

