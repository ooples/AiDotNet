---
title: "ISafetyModule<T>"
description: "Base interface for all safety modules in the composable safety pipeline."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Base interface for all safety modules in the composable safety pipeline.

## For Beginners

Each safety module is like a specialist inspector. One checks for
toxic language, another for NSFW images, another for PII, etc. You assemble the inspectors
you need into a pipeline, and the pipeline runs them all and gives you a combined report.

## How It Works

Safety modules are composable units that each check for a specific type of safety risk.
Multiple modules are combined into a `SafetyPipeline` that
runs them in sequence and aggregates their findings into a single `SafetyReport`.

**Architecture:** This replaces the monolithic ISafetyFilter with composable modules:

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` | Gets whether this module is ready to evaluate content. |
| `ModuleName` | Gets the unique name of this safety module. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` | Evaluates the given content for safety and returns any findings. |

