---
title: "JailbreakDetectorConfig"
description: "Configuration for jailbreak and prompt injection detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Text`

Configuration for jailbreak and prompt injection detection modules.

## For Beginners

Use this to configure how aggressively the jailbreak detector
works. Higher sensitivity catches more attack attempts but may also flag legitimate
prompts. Lower sensitivity is more lenient.

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectCharacterInjection` | Whether to check for character injection (emoji/Unicode smuggling). |
| `DetectEncodingAttacks` | Whether to check for encoding attacks (Base64, ROT13, Unicode). |
| `DetectMultiTurnAttacks` | Whether to check for multi-turn escalation attacks. |
| `Sensitivity` | Detection sensitivity (0.0 = lenient, 1.0 = strict). |

