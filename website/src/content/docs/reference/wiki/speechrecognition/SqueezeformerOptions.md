---
title: "SqueezeformerOptions"
description: "Configuration options for the Squeezeformer speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SpeechRecognition.ConformerFamily`

Configuration options for the Squeezeformer speech recognition model.

## How It Works

Squeezeformer (Kim et al., 2022) improves Conformer with a temporal U-Net structure,
micro-macro design (pre-norm instead of post-norm), and depthwise separable downsampling
for efficient computation with better accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SqueezeformerOptions` | Initializes a new instance with default values. |
| `SqueezeformerOptions(SqueezeformerOptions)` | Initializes a new instance by copying from another instance. |

