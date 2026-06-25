---
title: "EBranchformerOptions"
description: "Configuration options for the E-Branchformer (Enhanced Branchformer) speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SpeechRecognition.ConformerFamily`

Configuration options for the E-Branchformer (Enhanced Branchformer) speech recognition model.

## How It Works

E-Branchformer (Kim et al., 2022) improves on Branchformer with an enhanced merge module
that uses depthwise convolution for better local-global fusion, achieving SOTA on LibriSpeech
(WER 2.1%/4.2% test-clean/other) with the ESPnet toolkit.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EBranchformerOptions` | Initializes a new instance with default values. |
| `EBranchformerOptions(EBranchformerOptions)` | Initializes a new instance by copying from another instance. |

