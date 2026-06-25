---
title: "AST<T>"
description: "AST (Audio Spectrogram Transformer) model for audio event detection and classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

AST (Audio Spectrogram Transformer) model for audio event detection and classification.

## For Beginners

AST is one of the simplest yet effective audio classification models.
It works by treating sound spectrograms exactly like images and using a powerful image model
(Vision Transformer) to classify them.

Here is how AST processes audio, step by step:

**Step 1 - Sound to picture:** Audio is converted to a spectrogram (a 2D image showing
frequency vs time). This is the same mel spectrogram used by BEATs and other audio models.

**Step 2 - Cut into overlapping tiles:** The spectrogram is cut into small 16x16 patches
with overlap (stride 10), which means each patch shares some pixels with its neighbors.
This overlap improves accuracy at the cost of more patches to process.

**Step 3 - Add a special token:** A special [CLS] (classification) token is added.
This token will collect information from all patches and serve as the overall summary.

**Step 4 - Understand context:** A 12-layer Transformer encoder lets every patch attend
to every other patch. The [CLS] token gathers global information.

**Step 5 - Classify:** The [CLS] token output goes through a linear layer to produce
probabilities for each sound class. Sigmoid activation enables multi-label detection.

**Usage with a pre-trained ONNX model (recommended):****Usage with native training:**

## How It Works

AST (Gong et al., Interspeech 2021) is the first purely attention-based model for audio
classification. It directly applies a Vision Transformer (ViT) architecture to audio spectrograms,
achieving strong results on multiple benchmarks:

- **AudioSet**: 45.9% mAP - competitive with CNN-based models like PANNs
- **ESC-50**: 95.6% accuracy - environmental sound classification
- **Speech Commands V2**: 98.1% accuracy - keyword spotting

**Architecture:** AST treats the audio spectrogram as an image and processes it with
a standard Vision Transformer:

- **Audio preprocessing**: Raw waveform is converted to a 128-bin log-mel spectrogram
- **Patch embedding**: The spectrogram is split into overlapping 16x16 patches (stride 10),

each linearly projected to a 768-dimensional embedding

- **[CLS] token**: A learnable classification token is prepended to the patch sequence
- **Positional encoding**: Learnable positional embeddings encode spatial position
- **Transformer encoder**: 12-layer encoder with 12-head self-attention processes patches
- **Classification**: The [CLS] token output is projected to class logits via a linear head

**Key Innovation:** AST demonstrates that ImageNet-pretrained ViT weights transfer effectively
to audio spectrograms. By initializing from DeiT (Data-efficient Image Transformer) and fine-tuning
on AudioSet, AST achieves competitive results without any audio-specific architectural modifications.

**References:**

- Paper: "AST: Audio Spectrogram Transformer" (Gong et al., Interspeech 2021)
- Repository: https://github.com/YuanGongND/ast

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AST(NeuralNetworkArchitecture<>,ASTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an AST model for native training mode. |
| `AST(NeuralNetworkArchitecture<>,String,ASTOptions)` | Creates an AST model for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` | Gets the event labels. |
| `SupportedEvents` | Gets the list of event types this model can detect. |
| `TimeResolution` | Gets the time resolution for event detection in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(ASTOptions,IProgress<Double>,CancellationToken)` | Creates an AST model asynchronously with optional model download. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Detect(Tensor<>)` | Detects all audio events using the default confidence threshold. |
| `Detect(Tensor<>,)` | Detects audio events with a custom confidence threshold. |
| `DetectAsync(Tensor<>,CancellationToken)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` |  |
| `Dispose(Boolean)` |  |
| `GetEventProbabilities(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `StartStreamingSession` |  |
| `StartStreamingSession(Int32,)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `AudioSetLabels` | AudioSet-527 standard event labels used by AST pre-trained models. |

