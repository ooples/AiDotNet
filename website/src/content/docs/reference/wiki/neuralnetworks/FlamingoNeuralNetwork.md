---
title: "FlamingoNeuralNetwork<T>"
description: "Flamingo neural network for in-context visual learning and few-shot tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Flamingo neural network for in-context visual learning and few-shot tasks.

## For Beginners

Flamingo is a visual AI that can answer questions about images
with just a few examples. Show it 2-3 examples of image-answer pairs, and it can handle
similar questions about new images. It works by feeding image features into a language
model through special cross-attention layers, letting the language model "see" the image
while generating text responses.

## How It Works

Flamingo is a visual language model that excels at few-shot learning. It uses a Perceiver
Resampler to compress visual features and gated cross-attention layers to integrate
visual information into a frozen language model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlamingoNeuralNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,LanguageModelBackbone,Int32,ITokenizer,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,FlamingoOptions)` | Initializes a new instance using native layers. |
| `FlamingoNeuralNetwork(NeuralNetworkArchitecture<>,String,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,FlamingoOptions)` | Initializes a new instance using ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `ImageSize` |  |
| `LanguageModelBackbone` |  |
| `MaxImagesInContext` |  |
| `MaxSequenceLength` |  |
| `NumPerceiverTokens` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String,Int32)` |  |
| `ComputeImageTextSimilarity(Tensor<>,String)` |  |
| `ComputeSimilarity(Vector<>,Vector<>)` |  |
| `ConvertToTensor(Double[])` | Converts a double[] image to Tensor format. |
| `CreateNewInstance` |  |
| `DescribeVideo(IEnumerable<Tensor<>>,String,Int32)` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EncodeImage(Double[])` |  |
| `EncodeImageBatch(IEnumerable<Double[]>)` |  |
| `EncodeText(String)` |  |
| `EncodeTextBatch(IEnumerable<String>)` |  |
| `ExtractPerceiverFeatures(Tensor<>)` |  |
| `FewShotGenerate(IEnumerable<ValueTuple<Tensor<>,String>>,Tensor<>,String,Int32)` |  |
| `FewShotImageRetrieval(IEnumerable<Tensor<>>,String,IEnumerable<Tensor<>>,Int32)` |  |
| `FewShotVQA(IEnumerable<ValueTuple<Tensor<>,String,String>>,Tensor<>,String)` |  |
| `GenerateCaption(Tensor<>,Int32)` |  |
| `GenerateWithMultipleImages(IEnumerable<Tensor<>>,String,Int32)` |  |
| `GetImageEmbedding(Tensor<>)` |  |
| `GetImageEmbeddings(IEnumerable<Tensor<>>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `GetTextEmbedding(String)` |  |
| `GetTextEmbeddings(IEnumerable<String>)` |  |
| `InContextClassify(IEnumerable<ValueTuple<Tensor<>,String>>,Tensor<>)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `RetrieveImages(String,IEnumerable<Vector<>>,Int32)` |  |
| `RetrieveTexts(Tensor<>,IEnumerable<String>,Int32)` |  |
| `ScoreImageText(Tensor<>,String)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotClassify(Double[],IEnumerable<String>)` |  |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>)` |  |

