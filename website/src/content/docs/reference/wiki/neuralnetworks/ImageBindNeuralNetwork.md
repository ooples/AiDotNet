---
title: "ImageBindNeuralNetwork<T>"
description: "ImageBind neural network for binding multiple modalities (6+) into a shared embedding space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

ImageBind neural network for binding multiple modalities (6+) into a shared embedding space.

## For Beginners

ImageBind connects ALL types of data together!

Architecture overview:

1. Modality-Specific Encoders: Each modality has its own encoder (ViT for images, Transformer for text, etc.)
2. Projection Heads: Map each modality's features to the shared embedding space
3. Contrastive Learning: Align modalities using image as the bridge modality

Key capabilities:

- Cross-modal retrieval: Find images matching audio, text matching video, etc.
- Zero-shot classification: Classify any modality using text labels
- Emergent alignment: Compare modalities never directly paired during training

## How It Works

ImageBind learns a joint embedding space across multiple modalities: images, text, audio, depth,
thermal, and IMU data. It uses images as a binding modality - since web data contains
many (image, text) pairs, (image, audio) pairs from videos, etc., the model can learn
cross-modal relationships even without direct pairs between all modalities.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageBindNeuralNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,ITokenizer,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ImageBindOptions)` | Creates an ImageBind network using native library layers. |
| `ImageBindNeuralNetwork(NeuralNetworkArchitecture<>,String,String,String,ITokenizer,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ImageBindOptions)` | Creates an ImageBind network using pretrained ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `ParameterCount` |  |
| `SupportedModalities` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAlignment(ModalityType,Object,ModalityType,Object)` |  |
| `ComputeCrossModalSimilarity(Vector<>,Vector<>)` |  |
| `ComputeEmergentAudioTextSimilarity(Tensor<>,String)` |  |
| `CreateNewInstance` |  |
| `CrossModalRetrieval(Vector<>,IEnumerable<Vector<>>,Int32)` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `FindBestMatch(ModalityType,Object,IEnumerable<ValueTuple<ModalityType,Object>>)` |  |
| `FuseModalities(Dictionary<ModalityType,Vector<>>,String)` |  |
| `GenerateDescriptions(ModalityType,Object,IEnumerable<String>,Int32)` |  |
| `GetAudioEmbedding(Tensor<>,Int32)` |  |
| `GetDepthEmbedding(Tensor<>)` |  |
| `GetEmbedding(ModalityType,Object)` |  |
| `GetIMUEmbedding(Tensor<>)` |  |
| `GetImageEmbedding(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `GetTextEmbedding(String)` |  |
| `GetThermalEmbedding(Tensor<>)` |  |
| `GetVideoEmbedding(IEnumerable<Tensor<>>)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotClassify(ModalityType,Object,IEnumerable<String>)` |  |

