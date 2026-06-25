---
title: "FullSubNetPlusOptions"
description: "Configuration options for the FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Enhancement`

Configuration options for the FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) model.

## For Beginners

FullSubNet+ is a two-part neural network that cleans up noisy audio:

- The "full-band" part looks at all frequencies together to understand the overall pattern
- The "sub-band" part focuses on small frequency ranges for fine detail
- The two parts share information to get the best of both worlds

Think of it like cleaning a painting: one person looks at the whole picture for context,
while another works on details, and they coordinate together.

## How It Works

FullSubNet+ (Chen et al., ICASSP 2022) improves upon FullSubNet by using a channel-attention-based
full-band model and redesigned sub-band inputs. It achieves state-of-the-art speech enhancement
on the DNS Challenge dataset with PESQ 3.25 and STOI 0.96 at 16 kHz.

**References:**

- Paper: "FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms" (Chen et al., ICASSP 2022)
- Repository: https://github.com/hit-thusz-RookieCJ/FullSubNet-plus

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EnhancementStrength` | Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum). |
| `FftSize` | Gets or sets the FFT window size. |
| `FullBandHiddenSize` | Gets or sets the hidden size for full-band LSTM. |
| `FullBandLayers` | Gets or sets the number of full-band LSTM layers. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumFreqBins` | Gets or sets the number of frequency bins (FftSize / 2 + 1). |
| `NumNeighborSubBands` | Gets or sets the number of neighboring sub-bands for context. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `SubBandHiddenSize` | Gets or sets the hidden size for sub-band LSTM. |
| `SubBandLayers` | Gets or sets the number of sub-band LSTM layers. |
| `SubBandWidth` | Gets or sets the sub-band bandwidth (number of frequency bins per sub-band). |
| `UseChannelAttention` | Gets or sets whether to use channel attention in the full-band model. |
| `UseComplexSpectrogram` | Gets or sets whether to use complex spectrograms (magnitude + phase). |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

