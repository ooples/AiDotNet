---
title: "LLaVANeuralNetwork<T>"
description: "LLaVA (Large Language and Vision Assistant) neural network for visual instruction following."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

LLaVA (Large Language and Vision Assistant) neural network for visual instruction following.

## For Beginners

LLaVA is like giving eyes to ChatGPT!

Architecture overview:

1. Vision Encoder (CLIP ViT-L/14): Extracts image patch features
2. Projection Layer (MLP): Maps visual features to LLM's embedding space
3. Large Language Model (LLaMA/Vicuna): Generates text responses

Key capabilities:

- Visual conversations: "What's in this image?" followed by "What color is the car?"
- Visual reasoning: Understanding relationships, counting, spatial awareness
- Instruction following: "Describe this image as if you were a poet"
- Multi-turn dialogue: Context-aware conversations about images

## How It Works

LLaVA connects a vision encoder (CLIP ViT) with a large language model (LLaMA/Vicuna)
through a simple projection layer, enabling visual conversations and instruction following.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LLaVANeuralNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,LanguageModelBackbone,String,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LLaVAOptions)` | Creates a LLaVA network using native library layers. |
| `LLaVANeuralNetwork(NeuralNetworkArchitecture<>,String,String,ITokenizer,LanguageModelBackbone,String,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LLaVAOptions)` | Creates a LLaVA network using pretrained ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `ImageSize` |  |
| `LanguageModelBackbone` |  |
| `MaxSequenceLength` |  |
| `NumVisualTokens` |  |
| `ParameterCount` |  |
| `VisionEncoderType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Chat(Tensor<>,IEnumerable<ValueTuple<String,String>>,String,Int32,Double)` |  |
| `CompareImages(Tensor<>,Tensor<>,IEnumerable<String>)` |  |
| `ComputeSimilarity(Vector<>,Vector<>)` |  |
| `ConvertToTensor(Double[])` | Converts a double[] image to Tensor format. |
| `CreateNewInstance` |  |
| `DescribeRegions(Tensor<>,IEnumerable<Vector<>>)` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeImage(Double[])` |  |
| `EncodeImageBatch(IEnumerable<Double[]>)` |  |
| `EncodeText(String)` |  |
| `EncodeTextBatch(IEnumerable<String>)` |  |
| `ExtractVisualFeatures(Tensor<>)` |  |
| `Generate(Tensor<>,String,Int32,Double,Double)` |  |
| `GenerateMultiple(Tensor<>,String,Int32,Double)` |  |
| `GetImageEmbedding(Tensor<>)` |  |
| `GetImageEmbeddings(IEnumerable<Tensor<>>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetTextEmbedding(String)` |  |
| `GetTextEmbeddings(IEnumerable<String>)` |  |
| `GroundObject(Tensor<>,String)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `ProjectToLanguageSpace(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotClassify(Double[],IEnumerable<String>)` |  |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>)` |  |

