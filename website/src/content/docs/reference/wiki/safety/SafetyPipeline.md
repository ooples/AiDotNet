---
title: "SafetyPipeline<T>"
description: "Composable safety pipeline that runs multiple safety modules and aggregates their findings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety`

Composable safety pipeline that runs multiple safety modules and aggregates their findings.

## For Beginners

Think of the safety pipeline as an assembly line of inspectors.
Content flows through each inspector (module), and at the end you get a combined report
of everything they found. If anything dangerous is detected, the pipeline can block
the content or throw an exception, depending on your configuration.

## How It Works

The SafetyPipeline is the runtime orchestrator for content safety. It holds a list of
`ISafetyModule` instances, runs them against content, and produces a
unified `SafetyReport`. It respects the `SafetyConfig` to
determine enabled modules, action thresholds, and exception behavior.

**Usage via facade:**
The pipeline is constructed automatically based on your SafetyConfig.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SafetyPipeline(SafetyConfig)` | Initializes a new safety pipeline with the given configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the configuration for this pipeline. |
| `Modules` | Gets the registered safety modules. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddModule(ISafetyModule<>)` | Adds a safety module to the pipeline. |
| `EnforcePolicy(SafetyReport,Boolean)` | Checks a safety report and throws `SafetyViolationException` if the content should be blocked based on the current configuration. |
| `EvaluateAudio(Vector<>,Int32)` | Evaluates audio content through all registered audio safety modules. |
| `EvaluateImage(Tensor<>)` | Evaluates image content through all registered image safety modules. |
| `EvaluateText(String)` | Evaluates text content through all registered text safety modules. |
| `EvaluateVector(Vector<>)` | Evaluates content represented as a numeric vector through all registered modules. |
| `EvaluateVideo(IReadOnlyList<Tensor<>>,Double)` | Evaluates video content through all registered video safety modules. |

