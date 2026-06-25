---
title: "SafetyPipelineBuilder<T>"
description: "Fluent builder for constructing a `SafetyPipeline` with custom modules."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Safety`

Fluent builder for constructing a `SafetyPipeline` with custom modules.

## For Beginners

This is an advanced builder for when you want to hand-pick exactly
which safety modules to use. Most users should use `ConfigureSafety()` on the
AiModelBuilder instead, which automatically sets up the right modules.

## How It Works

The SafetyPipelineBuilder provides a fluent API for manually assembling a safety pipeline
with specific modules. For most users, `SafetyPipelineFactory` with a
`SafetyConfig` is simpler. Use this builder when you need fine-grained control
over which modules are included and their order.

**Example:**

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAudioModule(IAudioSafetyModule<>)` | Adds an audio safety module to the pipeline. |
| `AddImageModule(IImageSafetyModule<>)` | Adds an image safety module to the pipeline. |
| `AddModule(ISafetyModule<>)` | Adds a safety module to the pipeline. |
| `AddModules(IEnumerable<ISafetyModule<>>)` | Adds multiple safety modules to the pipeline. |
| `AddTextModule(ITextSafetyModule<>)` | Adds a text safety module to the pipeline. |
| `AddVideoModule(IVideoSafetyModule<>)` | Adds a video safety module to the pipeline. |
| `Build` | Builds the safety pipeline with the configured modules. |
| `Configure(Action<SafetyConfig>)` | Configures the safety settings using an action delegate. |
| `WithConfig(SafetyConfig)` | Sets the safety configuration for the pipeline. |

