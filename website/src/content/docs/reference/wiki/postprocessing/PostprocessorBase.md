---
title: "PostprocessorBase<T, TInput, TOutput>"
description: "Abstract base class for all postprocessors providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Postprocessing`

Abstract base class for all postprocessors providing common functionality.

## For Beginners

This is the foundation that all postprocessors build on.
It provides common features like:

- Configuration management
- Batch processing support
- Error handling

When creating a new postprocessor, you extend this class and implement the abstract methods.

## How It Works

This class provides the template method pattern for postprocessing.
Derived classes implement the core processing logic while this base class
handles validation, configuration, and common operations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PostprocessorBase` | Creates a new instance of the postprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsConfigured` | Gets whether this postprocessor is configured and ready to use. |
| `NumOps` | Gets the numeric operations helper for type T. |
| `Settings` | Gets the configuration settings for this postprocessor. |
| `SupportsInverse` | Gets whether this postprocessor supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Configure(Dictionary<String,Object>)` | Configures the postprocessor with optional settings. |
| `ConfigureCore(Dictionary<String,Object>)` | Core configuration implementation. |
| `EnsureConfigured` | Ensures the postprocessor is configured before use. |
| `GetSetting(String,)` | Gets a setting value with type conversion. |
| `Inverse()` | Reverses the postprocessing (if supported). |
| `InverseCore()` | Core inverse transformation implementation. |
| `Process()` | Transforms model output into the final result format. |
| `ProcessBatch(IEnumerable<>)` | Transforms a batch of model outputs. |
| `ProcessCore()` | Core processing implementation. |
| `ValidateInput()` | Validates input before processing. |

