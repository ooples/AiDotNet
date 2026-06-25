---
title: "NeMoMultitask<T>"
description: "NeMo Multitask: NVIDIA NeMo AED model for ASR + translation + language identification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

NeMo Multitask: NVIDIA NeMo AED model for ASR + translation + language identification.

## For Beginners

NeMo Multitask uses an attention-based encoder-decoder (AED) architecture for multi-task speech processing. The Fast Conformer encoder processes mel spectrograms, and a Transformer decoder with cross-attention generates text conditioned on special...

## How It Works

**References:**

- Model: "NeMo Multitask AED" (NVIDIA, 2024)

NeMo Multitask uses an attention-based encoder-decoder (AED) architecture for multi-task
speech processing. The Fast Conformer encoder processes mel spectrograms, and a Transformer
decoder with cross-attention generates text conditioned on special task tokens: language ID,
task type (transcribe/translate/identify), and output language. The model is trained jointly
on ASR, translation, and language identification tasks using a shared encoder and separate
task-specific decoder prompts.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the multitask AED pipeline. |

