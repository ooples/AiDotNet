---
title: "OpenCLIP<T>"
description: "OpenCLIP (Open Contrastive Language-Image Pre-training) model for zero-shot classification and cross-modal retrieval using open-source training data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

OpenCLIP (Open Contrastive Language-Image Pre-training) model for zero-shot classification
and cross-modal retrieval using open-source training data.

## For Beginners

OpenCLIP works exactly like CLIP - it understands both images and text.
You give it an image and some text labels, and it tells you which label best matches the image,
without any additional training. The "Open" in OpenCLIP means it was trained on publicly available
data, so anyone can reproduce and verify the results.

**Usage:**

## How It Works

OpenCLIP (Ilharco et al., 2021) is an open-source reproduction and extension of OpenAI's CLIP,
trained on publicly available datasets like LAION-2B (2 billion image-text pairs) and LAION-5B
(5 billion pairs). It reproduces CLIP's dual-encoder architecture with a Vision Transformer (ViT)
for images and a text transformer for text, both projecting into a shared embedding space.

**Architecture:**

- **Vision encoder**: ViT-B/32 through ViT-bigG/14, processing images as patch sequences
- **Text encoder**: GPT-2-style transformer with causal attention mask
- **Projection heads**: Linear projections mapping both modalities to shared 512/768/1024-dim space
- **Contrastive loss**: InfoNCE with learnable temperature (symmetric cross-entropy)

**Key Innovation:** OpenCLIP demonstrates that CLIP's training recipe works effectively with
public data. Models trained on LAION-2B match or exceed OpenAI CLIP performance, with the largest
ViT-bigG/14 variant achieving 80.1% zero-shot accuracy on ImageNet.

**References:**

- Paper: "Reproducible Scaling Laws for Contrastive Language-Image Learning" (Cherti et al., CVPR 2023)
- Repository: https://github.com/mlfoundations/open_clip

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenCLIP(NeuralNetworkArchitecture<>,OpenCLIPOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an OpenCLIP model in native training mode. |
| `OpenCLIP(NeuralNetworkArchitecture<>,String,OpenCLIPOptions)` | Creates an OpenCLIP model in ONNX inference mode from a pre-trained model file. |

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
| `ComputeSimilarity(Tensor<>,String)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeImage(Tensor<>)` |  |
| `EncodeText(String)` |  |
| `EncodeTexts(String[])` |  |
| `GetExtraTrainableLayers` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessImage(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SyncImageSizeWithArchitecture` | Aligns `_options.ImageSize` with `Architecture.InputHeight` when the architecture declares an explicit square spatial extent. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotClassify(Tensor<>,String[])` |  |

