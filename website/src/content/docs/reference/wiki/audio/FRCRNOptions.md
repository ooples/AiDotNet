---
title: "FRCRNOptions"
description: "Configuration options for the FRCRN (Frequency Recurrence CRN) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Enhancement`

Configuration options for the FRCRN (Frequency Recurrence CRN) model.

## For Beginners

FRCRN uses a clever trick: it processes audio frequencies in
sequence (low to high) like reading a book, so each frequency knows about its neighbors.
This helps it distinguish speech from noise because speech frequencies are correlated
(they appear together), while noise frequencies are more random.

## How It Works

FRCRN (Zhao et al., ICASSP 2022, Alibaba DAMO Academy) uses frequency recurrence to
model spectral correlations and complex spectral mapping. It won 1st place in the
ICASSP 2022 DNS Challenge non-personalized track with superior noise suppression while
preserving speech quality.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderChannels` | Gets or sets the encoder channel dimension. |
| `FFTSize` | Gets or sets the FFT size. |
| `HopLength` | Gets or sets the hop length. |
| `LearningRate` | Gets or sets the learning rate. |
| `LstmHiddenSize` | Gets or sets the LSTM hidden size for frequency recurrence. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFreqBins` | Gets or sets the number of frequency bins. |
| `NumStages` | Gets or sets the number of encoder-decoder stages. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("base" or "large"). |

