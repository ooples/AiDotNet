---
title: "SigLIP2<T>"
description: "SigLIP 2 (Multilingual Vision-Language Encoders with Improved Semantic Understanding) for contrastive encoding, zero-shot classification, and captioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

SigLIP 2 (Multilingual Vision-Language Encoders with Improved Semantic Understanding)
for contrastive encoding, zero-shot classification, and captioning.

## For Beginners

SigLIP 2 improves on SigLIP by learning from three tasks at once:
matching images with text, describing what's in images, and filling in missing parts of images.
This multi-task approach produces better visual representations. The model also supports
many languages, making it useful for global applications.

**Usage:**

## How It Works

SigLIP 2 (Tschannen et al., 2025) extends SigLIP with a multi-objective training framework
that combines three losses: (1) sigmoid contrastive loss for image-text alignment, (2) an
autoregressive captioning loss via a lightweight decoder for semantic understanding, and
(3) a self-supervised masked image modeling (MIM) loss for spatial understanding. The model
supports 32+ languages through an extended multilingual vocabulary.

**Architecture:**

- **Vision encoder**: ViT with patch sizes 14 or 16, supporting 256-512px images
- **Text encoder**: Transformer with multilingual SentencePiece tokenization (250K vocab)
- **Captioning decoder**: Lightweight 4-layer autoregressive transformer that attends

to vision encoder features via cross-attention, trained with next-token prediction

- **MIM decoder**: 2-layer MLP decoder that predicts masked patch features from

unmasked patch embeddings

- **Sigmoid contrastive loss**: Per-pair binary classification as in SigLIP

**Key Innovation:** By training with multiple objectives simultaneously, SigLIP 2 produces
vision encoders with richer semantic understanding than contrastive-only training. The captioning
objective forces the encoder to capture fine-grained details, while MIM improves spatial awareness.
Online data curation dynamically adjusts the training data mixture for better representation quality.

**References:**

- Paper: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding"

(Tschannen et al., 2025)

- Original SigLIP: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., ICCV 2023)
- Repository: https://github.com/google-research/big_vision

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SigLIP2(NeuralNetworkArchitecture<>,SigLIP2Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SigLIP 2 model in native training mode. |
| `SigLIP2(NeuralNetworkArchitecture<>,String,SigLIP2Options)` | Creates a SigLIP 2 model in ONNX inference mode from a pre-trained model file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AiDotNet#VisionLanguage#Interfaces#IVisualEncoder{T}#ImageChannels` |  |
| `AiDotNet#VisionLanguage#Interfaces#IVisualEncoder{T}#ImageSize` |  |
| `EmbeddingDimension` |  |
| `MaxSequenceLength` |  |
| `ProjectionDimension` |  |
| `Temperature` |  |
| `TextEmbeddingDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLayerBoundaries` | Computes the layer boundary indices for vision encoder, text encoder, captioning decoder, and MIM decoder sections. |
| `ComputeSimilarity(Tensor<>,String)` |  |
| `CreateNewInstance` |  |
| `CrossAttend(Tensor<>,Tensor<>)` | Cross-attention: decoder query attends to vision encoder key/value features. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeImage(Tensor<>)` |  |
| `EncodeText(String)` |  |
| `EncodeTexts(String[])` |  |
| `ForwardCaptioningDecoder(Tensor<>)` | Forwards through the captioning decoder layers. |
| `ForwardForTraining(Tensor<>)` |  |
| `ForwardMimDecoder(Tensor<>)` | Forwards through the MIM decoder layers to predict masked patch features. |
| `ForwardTextEncoder(Tensor<>)` | Forwards input through the text encoder layers. |
| `ForwardVisionEncoder(Tensor<>)` | Forwards input through the vision encoder layers. |
| `GenerateCaption(Tensor<>)` | Generates a caption for the given image using the autoregressive captioning decoder. |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PredictMaskedPatches(Tensor<>,Int32[])` | Computes masked image modeling predictions for self-supervised training. |
| `PreprocessImage(Tensor<>)` |  |
| `ProjectText(Tensor<>)` | Projects text encoder output to the shared embedding space. |
| `ProjectVision(Tensor<>)` | Projects vision encoder output to the shared embedding space. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotClassify(Tensor<>,String[])` |  |

