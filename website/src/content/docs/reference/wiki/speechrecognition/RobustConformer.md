---
title: "RobustConformer<T>"
description: "Robust Conformer: adversarially-trained Conformer for noisy ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Robust`

Robust Conformer: adversarially-trained Conformer for noisy ASR

## For Beginners

Robust Conformer uses adversarial training and speech reconstruction objectives to improve noise robustness. The model adds a speech reconstruction head during training that forces the encoder to preserve clean speech information even when process...

## How It Works

**References:**

- Paper: "Improving Noise Robustness of Contrastive Speech Representation Learning with Speech Reconstruction" (2023)

Robust Conformer uses adversarial training and speech reconstruction objectives to improve noise robustness. The model adds a speech reconstruction head during training that forces the encoder to preserve clean speech information even when processing noisy input. An adversarial noise classifier encourages the encoder to produce noise-invariant representations. The dual objective produces a Conformer encoder that maintains accuracy across clean and noisy conditions without explicit noise estimation.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using adversarially-trained Conformer encoder with CTC. |

