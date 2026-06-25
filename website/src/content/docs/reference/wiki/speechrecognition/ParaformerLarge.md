---
title: "ParaformerLarge<T>"
description: "Paraformer-Large: 220M parameter version for production ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

Paraformer-Large: 220M parameter version for production ASR

## For Beginners

Paraformer-Large scales up the Paraformer architecture to 220M parameters with a wider encoder (1024-dim, 16 heads) and deeper stack (50 encoder layers). Trained on 60k hours of Mandarin/English data with data augmentation (SpecAugment, speed pert...

## How It Works

**References:**

- Model: "Paraformer-Large" (Alibaba DAMO/FunASR, 2023)

Paraformer-Large scales up the Paraformer architecture to 220M parameters with a wider encoder (1024-dim, 16 heads) and deeper stack (50 encoder layers). Trained on 60k hours of Mandarin/English data with data augmentation (SpecAugment, speed perturbation). The model uses a 4x convolutional subsampling front-end and SentencePiece tokenization. Achieves state-of-the-art CER on AISHELL-1 and competitive results on LibriSpeech.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the scaled Paraformer-Large CIF pipeline. |

