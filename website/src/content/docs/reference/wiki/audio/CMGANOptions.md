---
title: "CMGANOptions"
description: "Configuration options for the CMGAN (Conformer-based Metric GAN) speech enhancement model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Enhancement`

Configuration options for the CMGAN (Conformer-based Metric GAN) speech enhancement model.

## For Beginners

CMGAN uses a "competition" strategy to enhance speech:

- A "generator" network tries to clean up noisy audio
- A "discriminator" network judges how realistic the cleaned audio sounds
- They train together: the generator gets better at fooling the discriminator
- The result is very natural-sounding enhanced speech

The "Conformer" part means it uses a mix of attention (looking at the whole signal)
and convolution (looking at local patterns), getting the best of both approaches.

## How It Works

CMGAN (Cao et al., INTERSPEECH 2022) combines a conformer-based generator with a metric
discriminator for high-quality speech enhancement. It achieves PESQ 3.41 and STOI 0.97
on the VoiceBank-DEMAND dataset, outperforming previous GAN-based methods.

**References:**

- Paper: "CMGAN: Conformer-based Metric GAN for Speech Enhancement" (Cao et al., INTERSPEECH 2022)
- Repository: https://github.com/ruizhecao96/CMGAN

## Properties

| Property | Summary |
|:-----|:--------|
| `ConformerDim` | Gets or sets the Conformer hidden dimension. |
| `ConformerKernelSize` | Gets or sets the convolution kernel size in Conformer layers. |
| `DecoderKernelSize` | Gets or sets the decoder kernel size. |
| `DiscriminatorLearningRate` | Gets or sets the discriminator learning rate. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderChannels` | Gets or sets the number of U-Net encoder channels. |
| `EnhancePhase` | Gets or sets whether to enhance both magnitude and phase. |
| `EnhancementStrength` | Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum). |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the generator learning rate. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in Conformer layers. |
| `NumConformerLayers` | Gets or sets the number of Conformer encoder layers. |
| `NumFreqBins` | Gets or sets the number of frequency bins. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `UseDiscriminator` | Gets or sets whether to use the metric discriminator during training. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

