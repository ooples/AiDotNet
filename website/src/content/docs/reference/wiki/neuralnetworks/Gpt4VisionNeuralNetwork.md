---
title: "Gpt4VisionNeuralNetwork<T>"
description: "GPT-4V-style neural network that combines vision understanding with large language model capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

GPT-4V-style neural network that combines vision understanding with large language model capabilities.

## For Beginners

GPT-4 Vision combines a visual understanding system with a
language model, letting you ask questions about images and receive natural language answers.
It first processes an image through a vision encoder (extracting features like objects,
text, and spatial relationships), then feeds those features into a language model that
generates human-readable responses. This enables tasks like describing images, reading
text in photos, and answering visual questions.

## How It Works

This implementation provides a vision-language model that can understand images and generate
text responses, similar to GPT-4V, LLaVA, or other vision-language models.

**Architecture Overview:**

1. Vision Encoder: ViT-based encoder to extract visual features
2. Vision-Language Projector: Maps visual features to LLM embedding space
3. Language Model: Transformer decoder for text generation
4. Multi-modal Attention: Allows text to attend to visual features

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Gpt4VisionNeuralNetwork(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,ILossFunction<>,Gpt4VisionOptions)` | Creates a GPT-4 Vision network using native layers (for training or when ONNX is not available). |
| `Gpt4VisionNeuralNetwork(NeuralNetworkArchitecture<>,String,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,ILossFunction<>,Gpt4VisionOptions)` | Creates a GPT-4 Vision network using pretrained ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextWindowSize` |  |
| `EmbeddingDimension` |  |
| `ImageEmbeddingDimension` |  |
| `ImageSize` |  |
| `MaxImageResolution` |  |
| `MaxImagesPerRequest` |  |
| `MaxSequenceLength` |  |
| `ParameterCount` |  |
| `SupportedDetailLevels` |  |
| `TextEmbeddingDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeChart(Tensor<>)` |  |
| `AnalyzeDocument(Tensor<>,String,String)` |  |
| `AnswerVisualQuestion(Tensor<>,String)` |  |
| `Chat(Tensor<>,IEnumerable<ValueTuple<String,String>>,String,Int32)` |  |
| `CompareImages(Tensor<>,Tensor<>,String)` |  |
| `ComputeSimilarity(Tensor<>,String)` |  |
| `ComputeSimilarity(Vector<>,Vector<>)` |  |
| `ConvertToTensor(Double[])` | Converts a double[] image to Tensor format. |
| `CreateNewInstance` |  |
| `DescribeImage(Tensor<>,String,String)` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectObjects(Tensor<>,String)` |  |
| `Dispose(Boolean)` |  |
| `EncodeImage(Double[])` |  |
| `EncodeImageBatch(IEnumerable<Double[]>)` |  |
| `EncodeText(String)` |  |
| `EncodeTextBatch(IEnumerable<String>)` |  |
| `EvaluateImageQuality(Tensor<>)` |  |
| `ExtractStructuredData(Tensor<>,String)` |  |
| `ExtractText(Tensor<>,Boolean)` |  |
| `Generate(Tensor<>,String,Int32,Double)` |  |
| `GenerateCodeFromUI(Tensor<>,String,String)` |  |
| `GenerateEditInstructions(Tensor<>,String)` |  |
| `GenerateFromMultipleImages(IEnumerable<Tensor<>>,String,Int32,Double)` |  |
| `GenerateStory(Tensor<>,String,String)` |  |
| `GetAttentionMap(Tensor<>,String)` |  |
| `GetGpt4VParameterGradients` | Gets the gradients for all trainable parameters. |
| `GetImageEmbedding(Tensor<>)` |  |
| `GetImageEmbeddings(IEnumerable<Tensor<>>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetTextEmbedding(String)` |  |
| `GetTextEmbeddings(IEnumerable<String>)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `RetrieveImages(String,IEnumerable<Tensor<>>,Int32)` |  |
| `RetrieveTexts(Tensor<>,IEnumerable<String>,Int32)` |  |
| `SafetyCheck(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `VisualReasoning(Tensor<>,String,String)` |  |
| `ZeroShotClassify(Double[],IEnumerable<String>)` |  |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |

