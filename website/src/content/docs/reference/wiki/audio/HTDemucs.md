---
title: "HTDemucs<T>"
description: "HTDemucs (Hybrid Transformer Demucs) for music source separation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SourceSeparation`

HTDemucs (Hybrid Transformer Demucs) for music source separation.

## For Beginners

HTDemucs is Meta's best music separator. It processes audio in two
ways at once: as raw sound waves and as a frequency picture (spectrogram). A Transformer
helps both paths share information, getting the best of both worlds.

**Usage:**

## How It Works

HTDemucs (Rouard et al., ICASSP 2023) is Meta's hybrid architecture combining a temporal
convolutional encoder with cross-domain Transformer attention, operating on both waveform
and spectrogram simultaneously. Achieves 9.0 dB SDR on MUSDB18-HQ.

