---
title: "SafetyConfig"
description: "Master configuration for the comprehensive safety pipeline."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Master configuration for the comprehensive safety pipeline.

## For Beginners

This is your one-stop safety control panel. You configure
everything through this single object:

All settings use nullable types with industry-standard defaults — if you don't set
something, a sensible default is used automatically.

## How It Works

This is the single configuration object for all safety features, accessed via
`AiModelBuilder.ConfigureSafety(Action<SafetyConfig>)`. It contains nested
sub-configs for each safety domain (text, image, audio, video, watermarking,
guardrails, fairness, compliance).

## Properties

| Property | Summary |
|:-----|:--------|
| `Audio` | Gets the audio safety configuration (deepfake, toxic speech, voice protection). |
| `Compliance` | Gets the regulatory compliance configuration (EU AI Act, GDPR, SOC2). |
| `DefaultAction` | Gets or sets the default action for safety violations when no module-specific action is configured. |
| `Enabled` | Gets or sets whether safety is enabled globally. |
| `Fairness` | Gets the fairness configuration (bias detection, demographic parity). |
| `Guardrails` | Gets the guardrails configuration (input/output guardrails, topic restrictions). |
| `Image` | Gets the image safety configuration (NSFW, violence, deepfake). |
| `MinimumActionSeverity` | Gets or sets the minimum severity level that triggers the configured action. |
| `Text` | Gets the text safety configuration (toxicity, PII, jailbreak, hallucination, copyright). |
| `ThrowOnUnsafeInput` | Gets or sets whether to throw an exception when unsafe input is detected. |
| `ThrowOnUnsafeOutput` | Gets or sets whether to throw an exception when unsafe output is detected. |
| `Video` | Gets the video safety configuration (content moderation, temporal deepfake). |
| `Watermarking` | Gets the watermarking configuration (text, image, audio watermarking). |

