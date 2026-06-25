---
title: "IEndToEndTts<T>"
description: "Interface for end-to-end TTS models that generate waveforms directly from text without a separate vocoder stage."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TextToSpeech.Interfaces`

Interface for end-to-end TTS models that generate waveforms directly from text
without a separate vocoder stage.

## How It Works

End-to-end models combine acoustic modeling and waveform generation in a single model:
Text -> [Single Model] -> Waveform.
Architectures include:

- VITS: VAE + normalizing flow + adversarial training
- VITS2: improved alignment and duration prediction
- YourTTS: multilingual zero-shot VITS variant
- Piper: lightweight VITS for edge deployment

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimension of the model's internal representation. |
| `NumFlowSteps` | Gets the number of flow steps used in the posterior encoder (for VAE-based models). |

