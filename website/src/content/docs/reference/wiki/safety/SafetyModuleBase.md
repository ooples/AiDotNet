---
title: "SafetyModuleBase<T>"
description: "Abstract base class for all safety modules, providing common infrastructure."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety`

Abstract base class for all safety modules, providing common infrastructure.

## For Beginners

This is the common ancestor of all safety modules. It handles
boilerplate like module naming and readiness checks so each module can focus on
its specific detection logic.

## How It Works

Provides shared functionality for all safety modules including the module name,
readiness state, and default numeric operations. Concrete safety modules should
typically inherit from a more specific base class (TextSafetyModuleBase,
ImageSafetyModuleBase, etc.) rather than this one directly.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `Evaluate(Vector<>)` |  |
| `GetParameters` |  |
| `Predict(Vector<>)` | Predicts safety scores from content by delegating to Evaluate and converting findings to a score vector. |
| `SetParameters(Vector<>)` |  |
| `Train(Vector<>,Vector<>)` | Training is not typically used for safety modules. |
| `WithParameters(Vector<>)` |  |

