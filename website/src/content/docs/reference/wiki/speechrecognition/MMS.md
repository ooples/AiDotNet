---
title: "MMS<T>"
description: "MMS: Massively Multilingual Speech covering 1100+ languages"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Multilingual`

MMS: Massively Multilingual Speech covering 1100+ languages

## For Beginners

MMS (Massively Multilingual Speech) extends wav2vec 2.0 to 1100+ languages using religious text recordings and other multilingual data sources. The model uses a shared Transformer encoder pre-trained on unlabeled speech from 1400+ languages, then ...

## How It Works

**References:**

- Paper: "Scaling Speech Technology to 1,000+ Languages" (Pratap et al., Meta, 2023)

MMS (Massively Multilingual Speech) extends wav2vec 2.0 to 1100+ languages using religious text recordings and other multilingual data sources. The model uses a shared Transformer encoder pre-trained on unlabeled speech from 1400+ languages, then fine-tuned with CTC on labeled data. Language-specific adapter layers enable efficient multi-language support without full model duplication. Achieves strong ASR for many low-resource languages previously unsupported.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using MMS's multilingual SSL encoder with CTC. |

