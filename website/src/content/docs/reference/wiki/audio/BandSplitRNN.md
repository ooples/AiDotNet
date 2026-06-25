---
title: "BandSplitRNN<T>"
description: "BandSplitRNN for music source separation (Luo and Yu, 2023)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SourceSeparation`

BandSplitRNN for music source separation (Luo and Yu, 2023).

## For Beginners

BandSplitRNN separates a mixed song into individual instruments (vocals,
drums, bass, other) by dividing audio frequencies into bands—like splitting an equalizer into
sections. Each section is cleaned up by one specialist RNN, then a second RNN checks that the
sections work together consistently. The result is clean, separated audio for each instrument.

**Usage:**

## How It Works

BandSplitRNN splits the spectrogram into non-overlapping frequency bands, processes each band
independently with a shared band-level RNN, applies cross-band fusion via a sequence-level RNN,
and estimates source-specific masks. This dual-RNN design achieves 10.0+ dB SDR on MUSDB18-HQ,
establishing a strong RNN-based baseline for music source separation.

