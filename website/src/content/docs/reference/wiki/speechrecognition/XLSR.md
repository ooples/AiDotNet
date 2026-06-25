---
title: "XLSR<T>"
description: "XLS-R: cross-lingual speech representation at scale"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Multilingual`

XLS-R: cross-lingual speech representation at scale

## For Beginners

XLS-R scales wav2vec 2.0 pre-training to 128 languages using 436k hours of unlabeled speech. The model (up to 2B parameters) uses the same contrastive pre-training approach but with massively multilingual data. Fine-tuning with CTC on labeled data...

## How It Works

**References:**

- Paper: "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale" (Babu et al., Meta, 2022)

XLS-R scales wav2vec 2.0 pre-training to 128 languages using 436k hours of unlabeled speech. The model (up to 2B parameters) uses the same contrastive pre-training approach but with massively multilingual data. Fine-tuning with CTC on labeled data achieves strong results across many languages, with particularly significant improvements for low-resource languages. The cross-lingual transfer is enabled by shared phonetic structures across languages.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using XLS-R's cross-lingual SSL encoder with CTC. |

