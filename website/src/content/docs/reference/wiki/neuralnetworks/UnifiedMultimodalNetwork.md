---
title: "UnifiedMultimodalNetwork<T>"
description: "Unified multimodal network that handles text, images, audio, and video in a single architecture with cross-modal attention and any-to-any generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Unified multimodal network that handles text, images, audio, and video
in a single architecture with cross-modal attention and any-to-any generation.

## For Beginners

This network handles text, images, audio, and video all in
one architecture. Instead of needing separate models for each type of data, it processes
all modalities through shared transformer layers with cross-modal attention. It can both
understand any combination of inputs and generate outputs in any modality, enabling tasks
like describing a video, generating images from text, or answering questions about audio
clips.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UnifiedMultimodalNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Nullable<Int32>,UnifiedMultimodalNetworkOptions)` | Initializes a new instance of the UnifiedMultimodalNetwork. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `MaxSequenceLength` |  |
| `ParameterCount` |  |
| `SupportedInputModalities` |  |
| `SupportedOutputModalities` |  |
| `SupportsStreaming` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignTemporally(IEnumerable<MultimodalInput<>>)` |  |
| `AnswerQuestion(IEnumerable<MultimodalInput<>>,String)` |  |
| `Chat(IEnumerable<ValueTuple<String,IEnumerable<MultimodalInput<>>>>,IEnumerable<MultimodalInput<>>,Int32)` |  |
| `Compare(IEnumerable<MultimodalInput<>>,IEnumerable<String>)` |  |
| `ComputeSimilarity(MultimodalInput<>,MultimodalInput<>)` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Detect(IEnumerable<MultimodalInput<>>,String)` |  |
| `Edit(MultimodalInput<>,String)` |  |
| `Encode(MultimodalInput<>)` |  |
| `EncodeSequence(IEnumerable<MultimodalInput<>>)` |  |
| `FewShotLearn(IEnumerable<ValueTuple<IEnumerable<MultimodalInput<>>,MultimodalOutput<>>>,IEnumerable<MultimodalInput<>>)` |  |
| `ForwardForTraining(Tensor<>)` | Training-mode forward — same architecture as `Predict` but routes every reshape through `Int32[])` so the gradient tape stays connected end-to-end. |
| `Fuse(IEnumerable<MultimodalInput<>>,String)` |  |
| `Generate(IEnumerable<MultimodalInput<>>,ModalityType,Int32)` |  |
| `GenerateAudio(IEnumerable<MultimodalInput<>>,Double,Int32)` |  |
| `GenerateImage(IEnumerable<MultimodalInput<>>,Int32,Int32)` |  |
| `GenerateInterleaved(IEnumerable<MultimodalInput<>>,IEnumerable<ValueTuple<ModalityType,Int32>>)` |  |
| `GenerateText(IEnumerable<MultimodalInput<>>,String,Int32,Double)` |  |
| `GetCrossModalAttention(IEnumerable<MultimodalInput<>>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `Reason(IEnumerable<MultimodalInput<>>,String)` |  |
| `Retrieve(MultimodalInput<>,IEnumerable<MultimodalInput<>>,Int32)` |  |
| `SafetyCheck(IEnumerable<MultimodalInput<>>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetParameters(Vector<>)` |  |
| `ShapesMatch(Tensor<>,Tensor<>)` | True iff the two tensors are element-wise broadcastable without any reshape (same rank, same dim values). |
| `Summarize(IEnumerable<MultimodalInput<>>,ModalityType,Int32)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `Translate(MultimodalInput<>,ModalityType)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultOutputSize` | Initializes a new instance with default architecture settings. |

