---
title: "VideoCLIPNeuralNetwork<T>"
description: "VideoCLIP neural network for video-text alignment and temporal understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

VideoCLIP neural network for video-text alignment and temporal understanding.

## For Beginners

VideoCLIP is like CLIP but for videos!

Architecture overview:

1. Vision Encoder: Extracts features from each frame (shared CLIP ViT)
2. Temporal Encoder: Aggregates frame features over time
3. Text Encoder: Processes text descriptions
4. Contrastive Learning: Aligns video and text in shared embedding space

Key capabilities:

- Video retrieval: Find videos matching text descriptions
- Action recognition: Classify actions without training
- Moment localization: Find specific moments in videos
- Video QA: Answer questions about video content

## How It Works

VideoCLIP extends CLIP's contrastive learning paradigm to the video domain, enabling
text-to-video and video-to-text retrieval, action recognition, and temporal understanding.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoCLIPNeuralNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,TemporalAggregationType,ITokenizer,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,VideoCLIPOptions)` | Creates a VideoCLIP network using native library layers. |
| `VideoCLIPNeuralNetwork(NeuralNetworkArchitecture<>,String,String,ITokenizer,Int32,Double,TemporalAggregationType,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,VideoCLIPOptions)` | Creates a VideoCLIP network using pretrained ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `FrameRate` |  |
| `ImageSize` |  |
| `MaxSequenceLength` |  |
| `NumFrames` |  |
| `ParameterCount` |  |
| `TemporalAggregation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerVideoQuestion(IEnumerable<Tensor<>>,String,Int32)` |  |
| `CombineVideoTextContext(Tensor<>,Tensor<>)` | Combines video and text contexts for generation. |
| `ComputeSimilarity(Vector<>,Vector<>)` |  |
| `ComputeTemporalSimilarityMatrix(IEnumerable<Tensor<>>,IEnumerable<Tensor<>>)` |  |
| `ComputeVideoTextSimilarity(String,IEnumerable<Tensor<>>)` |  |
| `ConvertToTensor(Double[])` | Converts a double[] image to Tensor format. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Decodes token IDs back to text. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeFrameNative(Tensor<>)` | Legacy frame encoder returning Vector for non-training paths. |
| `EncodeFrameNativeTensor(Tensor<>)` | Encodes a single frame using Engine operations for tape-tracked gradient flow. |
| `EncodeImage(Double[])` |  |
| `EncodeImageBatch(IEnumerable<Double[]>)` |  |
| `EncodeText(String)` |  |
| `EncodeTextBatch(IEnumerable<String>)` |  |
| `EncodeTextNative(String)` | Legacy text encoder returning Vector for public API and non-training paths. |
| `EncodeTextNativeTensor(String)` | Encodes text as a normalized embedding tensor using tape-tracked Engine operations. |
| `EncodeVideoNative(List<Tensor<>>)` | Legacy video encoder returning Vector for public API and non-training paths. |
| `EncodeVideoNativeTensor(List<Tensor<>>)` | Encodes a video as a normalized embedding tensor using tape-tracked Engine operations. |
| `ExtractFrameFeatures(IEnumerable<Tensor<>>)` |  |
| `ForwardForTraining(Tensor<>)` |  |
| `GenerateAnswerAutoregressive(Vector<>,Vector<>,String,Int32)` | Generates an answer autoregressively using video and question context. |
| `GenerateCaptionAutoregressive(Vector<>,Int32)` | Generates a caption autoregressively token by token. |
| `GenerateVideoCaption(IEnumerable<Tensor<>>,Int32)` |  |
| `GetImageEmbedding(Tensor<>)` |  |
| `GetImageEmbeddings(IEnumerable<Tensor<>>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `GetSequenceEmbedding(List<Int32>)` | Gets embedding for a sequence of token IDs. |
| `GetTextEmbedding(String)` |  |
| `GetTextEmbeddings(IEnumerable<String>)` |  |
| `GetVideoEmbedding(IEnumerable<Tensor<>>)` |  |
| `GetVideoEmbeddings(IEnumerable<IEnumerable<Tensor<>>>)` |  |
| `InitializeLayers` |  |
| `IsYesNoQuestion(String)` | Determines if a question expects a yes/no answer. |
| `LocalizeMoments(IEnumerable<Tensor<>>,String,Int32)` |  |
| `MeanPool(Tensor<>)` | Legacy mean-pool returning Vector for non-training paths (ONNX, caption generation). |
| `MeanPoolTensor(Tensor<>)` | Mean-pools a 2D tensor [seqLen, hiddenDim] along axis 0, returning a 1D tensor [hiddenDim]. |
| `Normalize(Vector<>)` | Legacy normalize returning Vector for non-training paths. |
| `NormalizeTensor(Tensor<>)` | L2-normalizes a 1D tensor using Engine operations for tape-tracked gradient flow. |
| `ParseInputToFrames(Tensor<>)` | Parses a batched input tensor into a list of individual frame tensors. |
| `PredictCore(Tensor<>)` |  |
| `PredictNextAction(IEnumerable<Tensor<>>,IEnumerable<String>)` |  |
| `RetrieveTextsForVideo(IEnumerable<Tensor<>>,IEnumerable<String>,Int32)` |  |
| `RetrieveVideos(String,IEnumerable<Vector<>>,Int32)` |  |
| `SampleNextToken([],Double,Double)` | Samples the next token using temperature scaling and nucleus (top-p) sampling. |
| `SampleYesNoToken([])` | Samples a yes/no token with constrained decoding. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotActionRecognition(IEnumerable<Tensor<>>,IEnumerable<String>)` |  |
| `ZeroShotClassify(Double[],IEnumerable<String>)` |  |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>)` |  |

