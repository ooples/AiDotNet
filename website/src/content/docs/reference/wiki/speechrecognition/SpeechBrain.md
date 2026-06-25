---
title: "SpeechBrain<T>"
description: "SpeechBrain: open-source speech processing toolkit ASR models"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Robust`

SpeechBrain: open-source speech processing toolkit ASR models

## For Beginners

SpeechBrain provides a comprehensive set of pre-trained ASR models built with the open-source toolkit. The models use various architectures including CRDNN (convolutional-recurrent-DNN), Transformer, and Conformer encoders with CTC or attention-ba...

## How It Works

**References:**

- Paper: "SpeechBrain: A General-Purpose Speech Toolkit" (Ravanelli et al., 2021)

SpeechBrain provides a comprehensive set of pre-trained ASR models built with the open-source toolkit. The models use various architectures including CRDNN (convolutional-recurrent-DNN), Transformer, and Conformer encoders with CTC or attention-based decoders. SpeechBrain models are trained with extensive data augmentation, dynamic batching, and mixed-precision training. The toolkit supports over 100 speech processing recipes covering ASR, speaker recognition, speech separation, and more.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using SpeechBrain's Conformer encoder with CTC decoding. |

