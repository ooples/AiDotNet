---
title: "SigLIP<T>"
description: "SigLIP (Sigmoid Loss for Language-Image Pre-training) model for zero-shot classification and cross-modal retrieval with improved batch scaling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

SigLIP (Sigmoid Loss for Language-Image Pre-training) model for zero-shot classification
and cross-modal retrieval with improved batch scaling.

## For Beginners

SigLIP is an improved version of CLIP. The main difference is in how it
learns: instead of comparing every image with every text in a batch simultaneously (which requires
lots of memory), SigLIP looks at each image-text pair one at a time and asks "do these match?".
This simple change makes training faster and results better.

**Usage:**

## How It Works

SigLIP (Zhai et al., ICCV 2023) replaces CLIP's softmax-based InfoNCE loss with a sigmoid loss
that operates on individual image-text pairs independently. This removes the need for global
batch normalization, enabling efficient training with very large batch sizes (up to 1M pairs).
The model achieves 84.5% zero-shot on ImageNet with ViT-L/16@384.

**Architecture:**

- **Vision encoder**: ViT with patch sizes 16 or 14, supporting multiple image resolutions
- **Text encoder**: Transformer with causal attention, shared vocabulary
- **Sigmoid contrastive loss**: Per-pair binary classification: positive = 1, negative = -1
- **Learnable temperature + bias**: sigmoid(z * (sim/t + b)) where t and b are learned

**Key Innovation:** The sigmoid loss removes the softmax normalization across the full batch,
which means each image-text pair is treated independently. This allows efficient distributed training
without needing to gather all embeddings globally, and empirically gives better performance at scale.

**References:**

- Paper: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., ICCV 2023)
- Paper: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding" (2025)
- Repository: https://github.com/google-research/big_vision (SigLIP components)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SigLIP(NeuralNetworkArchitecture<>,SigLIPOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SigLIP model in native training mode. |
| `SigLIP(NeuralNetworkArchitecture<>,String,SigLIPOptions)` | Creates a SigLIP model in ONNX inference mode from a pre-trained model file. |

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

