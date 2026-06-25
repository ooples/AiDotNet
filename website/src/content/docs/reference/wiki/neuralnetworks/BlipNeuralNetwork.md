---
title: "BlipNeuralNetwork<T>"
description: "BLIP (Bootstrapped Language-Image Pre-training) neural network for vision-language tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

BLIP (Bootstrapped Language-Image Pre-training) neural network for vision-language tasks.

## For Beginners

BLIP is a more powerful version of CLIP!

CLIP can:

- Match images with text descriptions
- Zero-shot classification

BLIP adds:

- Generate captions ("a dog playing in the park")
- Answer questions ("What color is the car?" -> "Red")
- More accurate image-text matching

Training innovation:

- BLIP was trained on noisy web data
- It learned to filter out bad captions automatically
- Then it generated better captions to train on!
- This "bootstrapping" creates a cleaner dataset

Use cases:

- Accessibility (auto-generate alt-text for images)
- Content moderation (answer "is there violence in this image?")
- Visual search (find images matching a description)
- Image organization (auto-tag photos)

## How It Works

BLIP extends CLIP's capabilities with image captioning, image-text matching, and visual
question answering. It uses a unified framework with both understanding and generation tasks.
This implementation supports both ONNX pretrained models and native library layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BlipNeuralNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,ITokenizer,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,BlipOptions)` | Creates a BLIP network using native library layers. |
| `BlipNeuralNetwork(NeuralNetworkArchitecture<>,String,String,String,ITokenizer,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,BlipOptions)` | Creates a BLIP network using pretrained ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `ImageSize` |  |
| `MaxSequenceLength` |  |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String,Int32)` |  |
| `AnswerQuestionNative(Tensor<>,String,Int32)` | Answers a question using native layers. |
| `AnswerQuestionOnnx(Tensor<>,String,Int32)` | Answers question using ONNX. |
| `ArgMax(Tensor<>)` | Finds argmax of last dimension at the last sequence position for autoregressive decoding. |
| `ComputeImageTextMatch(Tensor<>,String)` |  |
| `ComputeImageTextMatchNative(Tensor<>,String)` | Computes ITM score using native layers. |
| `ComputeImageTextMatchOnnx(Tensor<>,String)` | Computes ITM score using ONNX. |
| `ComputeSimilarity(Vector<>,Vector<>)` |  |
| `ConvertToTensor(Double[])` | Converts a double[] image to Tensor format. |
| `CreateDefaultTokenizer` | Creates a default simple tokenizer for testing. |
| `CreateNewInstance` |  |
| `DecodeTokens(IEnumerable<Int32>)` | Decodes token IDs to text. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `EncodeImage(Double[])` |  |
| `EncodeImageBatch(IEnumerable<Double[]>)` |  |
| `EncodeText(String)` |  |
| `EncodeTextBatch(IEnumerable<String>)` |  |
| `ExtractEmbeddingFromOnnxTensor(Tensor<Single>)` | Extracts embedding from ONNX tensor. |
| `ForwardDecoderNative(Tensor<>,Tensor<>)` | Forward pass through decoder with image conditioning. |
| `GenerateCaption(Tensor<>,Int32,Int32)` |  |
| `GenerateCaptionNative(Tensor<>,Int32,Int32)` | Generates a caption using native layers. |
| `GenerateCaptionOnnx(Tensor<>,Int32,Int32)` | Generates caption using ONNX decoder. |
| `GenerateCaptionWithSampling(Tensor<>,Int32,Double)` | Generates caption with temperature sampling for diversity. |
| `GenerateCaptions(Tensor<>,Int32,Int32)` |  |
| `GetBosTokenId` | Gets the BOS token ID. |
| `GetEosTokenId` | Gets the EOS token ID. |
| `GetImageEmbedding(Tensor<>)` |  |
| `GetImageEmbeddingNative(Tensor<>)` | Gets image embedding using native layers. |
| `GetImageEmbeddingOnnx(Tensor<>)` | Gets image embedding using ONNX. |
| `GetImageEmbeddings(IEnumerable<Tensor<>>)` |  |
| `GetImageFeaturesNative(Tensor<>)` | Gets image features without final projection (for cross-attention). |
| `GetModelMetadata` | Retrieves metadata about the BLIP neural network model. |
| `GetOptions` |  |
| `GetParameters` |  |
| `GetTextEmbedding(String)` |  |
| `GetTextEmbeddingNative(String)` | Gets text embedding using native layers. |
| `GetTextEmbeddingOnnx(String)` | Gets text embedding using ONNX. |
| `GetTextEmbeddings(IEnumerable<String>)` |  |
| `GetTextFeaturesNative(String)` | Gets text features without final projection. |
| `InitializeLayers` |  |
| `InitializeNativeLayers` | Initializes native layers for the BLIP network. |
| `InitializeParameters` | Initialize parameters with small random values. |
| `LastPositionLogits(Tensor<>)` | Extracts the next-token logits (the last sequence position) from a decoder output tensor as a `Vector` for the shared `TokenSampler`. |
| `PredictCore(Tensor<>)` |  |
| `RankCaptions(Tensor<>,IEnumerable<String>)` |  |
| `RetrieveImages(String,IEnumerable<Vector<>>,Int32)` |  |
| `RetrieveTexts(Tensor<>,IEnumerable<Vector<>>,Int32)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotClassify(Double[],IEnumerable<String>)` |  |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_crossAttentionLayers` | Cross-attention layers for ITM (native mode). |
| `_embeddingDimension` | The dimensionality of the shared embedding space. |
| `_hiddenDim` | Hidden dimension for transformer layers. |
| `_imageSize` | Expected image size (width and height). |
| `_itmHead` | Image-Text Matching (ITM) classification head. |
| `_lmHead` | Language model head - projects hidden states to vocabulary logits. |
| `_lossFunction` | Loss function for training. |
| `_maxSequenceLength` | Maximum sequence length for text encoder. |
| `_mlpDim` | MLP hidden dimension. |
| `_numDecoderLayers` | Number of decoder layers. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer layers. |
| `_optimizer` | Optimizer for training. |
| `_patchEmbedding` | Patch embedding layer for vision. |
| `_patchSize` | Patch size for vision transformer. |
| `_textClsToken` | Learnable CLS token for text encoder. |
| `_textDecoder` | The ONNX inference session for the text decoder (for caption generation). |
| `_textDecoderLayers` | Text decoder layers for caption generation (native mode). |
| `_textDecoderPath` | Path to the text decoder ONNX model file (for ONNX mode). |
| `_textEncoder` | The ONNX inference session for the text encoder. |
| `_textEncoderLayers` | Text encoder layers (native mode). |
| `_textEncoderPath` | Path to the text encoder ONNX model file (for ONNX mode). |
| `_textPositionalEmbeddings` | Text positional embeddings. |
| `_textTokenEmbedding` | Text token embeddings (vocabulary lookup). |
| `_tokenizer` | The tokenizer for processing text input. |
| `_useNativeMode` | Indicates whether this BLIP network uses native layers (true) or ONNX models (false). |
| `_visionClsToken` | Learnable CLS token for vision encoder. |
| `_visionEncoder` | The ONNX inference session for the vision encoder. |
| `_visionEncoderLayers` | Vision transformer layers for image encoding (native mode). |
| `_visionEncoderPath` | Path to the vision encoder ONNX model file (for ONNX mode). |
| `_visionPositionalEmbeddings` | Vision positional embeddings. |
| `_vocabularySize` | Vocabulary size for text encoder. |

