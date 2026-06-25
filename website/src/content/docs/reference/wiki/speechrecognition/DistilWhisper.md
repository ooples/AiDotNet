---
title: "DistilWhisper<T>"
description: "Distil-Whisper: HuggingFace's knowledge-distilled Whisper, 756M params, 6x faster."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

Distil-Whisper: HuggingFace's knowledge-distilled Whisper, 756M params, 6x faster.

## For Beginners

Distil-Whisper uses layer-wise knowledge distillation from Whisper large-v2/v3. The student model keeps the full 32-layer encoder but reduces the decoder to 2 layers, trained with KL-divergence loss on pseudo-labeled data. The model achieves withi...

## How It Works

**References:**

- Paper: "Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling" (Gandhi et al., HuggingFace, 2023)

Distil-Whisper uses layer-wise knowledge distillation from Whisper large-v2/v3.
The student model keeps the full 32-layer encoder but reduces the decoder to 2 layers,
trained with KL-divergence loss on pseudo-labeled data. The model achieves within 1% WER
of the teacher on out-of-distribution test sets while being 6x faster. Key innovation:
using pseudo-labels from the teacher on large unlabeled audio corpora instead of the
original training data, which is unavailable.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the distilled encoder-decoder pipeline. |

