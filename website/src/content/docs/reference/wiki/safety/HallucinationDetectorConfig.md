---
title: "HallucinationDetectorConfig"
description: "Configuration for hallucination detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Text`

Configuration for hallucination detection modules.

## For Beginners

Use this to configure hallucination detection. You can set
the threshold for how confident the detector must be before flagging content as
hallucinated, and choose whether to use reference-based or self-consistency methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConsistencySamples` | Number of samples for self-consistency checking. |
| `PerClaimVerification` | Whether to extract and verify individual claims. |
| `Threshold` | Hallucination score threshold (0.0-1.0). |

