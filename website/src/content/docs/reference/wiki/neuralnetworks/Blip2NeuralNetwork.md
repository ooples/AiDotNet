---
title: "Blip2NeuralNetwork<T>"
description: "BLIP-2 (Bootstrapped Language-Image Pre-training 2) neural network for vision-language tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

BLIP-2 (Bootstrapped Language-Image Pre-training 2) neural network for vision-language tasks.

## For Beginners

BLIP-2 is the next evolution of vision-language models!

Architecture overview:

1. Frozen Image Encoder (ViT-G): Extracts image patch features
2. Q-Former: Small trainable transformer that bridges vision and language
- Uses 32 learnable "query" tokens
- Queries attend to image features via cross-attention
- Output goes to the language model
3. Frozen LLM (OPT/Flan-T5): Generates text from visual features

Why this architecture is brilliant:

- Only trains the small Q-Former (~188M parameters)
- Image encoder stays frozen (no GPU memory for gradients)
- LLM stays frozen (can use huge 66B+ models)
- Much cheaper to train than end-to-end models

Training stages:

1. Vision-Language Representation Learning (Q-Former + ViT)
2. Vision-to-Language Generative Learning (Q-Former + LLM)

## How It Works

BLIP-2 uses a Q-Former (Querying Transformer) to efficiently bridge frozen image encoders
with frozen large language models. The Q-Former uses learnable query tokens that interact
with frozen image features through cross-attention layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Blip2NeuralNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,LanguageModelBackbone,ITokenizer,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Blip2Options)` | Creates a BLIP-2 network using native library layers. |
| `Blip2NeuralNetwork(NeuralNetworkArchitecture<>,String,String,String,ITokenizer,LanguageModelBackbone,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Blip2Options)` | Creates a BLIP-2 network using pretrained ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `ImageSize` |  |
| `LanguageModelBackbone` |  |
| `MaxSequenceLength` |  |
| `NumQueryTokens` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateQueryTokenGradients(Tensor<>)` | Accumulates gradients for query tokens and positional embeddings from the backward pass. |
| `AnswerQuestion(Tensor<>,String,Int32)` |  |
| `ApplyCrossAttention(ILayer<>,Tensor<>,Tensor<>)` | Applies cross-attention between queries and keys/values. |
| `AttentionToBoundingBox(Tensor<>)` | Converts attention weights to bounding box. |
| `ComputeContrastiveSimilarity(Tensor<>,String)` |  |
| `ComputeImageTextMatch(Tensor<>,String)` |  |
| `ComputeItmNative(Tensor<>,String)` | Computes ITM score using native layers. |
| `ComputeItmOnnx(Tensor<>,String)` | Computes ITM score using ONNX models. |
| `ComputeSimilarity(Vector<>,Vector<>)` |  |
| `ConvertToTensor(Double[])` | Converts a double[] image to Tensor format. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `EncodeImage(Double[])` |  |
| `EncodeImageBatch(IEnumerable<Double[]>)` |  |
| `EncodeText(String)` |  |
| `EncodeTextBatch(IEnumerable<String>)` |  |
| `ExtractQFormerFeatures(Tensor<>)` |  |
| `ExtractQFormerFeaturesNative(Tensor<>)` | Extracts Q-Former features using native layers. |
| `ExtractQFormerFeaturesOnnx(Tensor<>)` | Extracts Q-Former features using ONNX models. |
| `Forward(Tensor<>)` | Forward pass through Q-Former and vision encoder. |
| `GenerateCaption(Tensor<>,String,Int32,Int32,Double)` |  |
| `GenerateCaptions(Tensor<>,Int32,String,Int32,Double,Double)` |  |
| `GenerateNative(Tensor<>,String,Int32,Double)` | Generates text using native layers with autoregressive decoding. |
| `GenerateOnnx(Tensor<>,String,Int32)` | Generates text using ONNX language model. |
| `GenerateWithInstruction(Tensor<>,String,Int32)` |  |
| `GenerateWithLm(Tensor<>,String,Int32,Int32,Double)` | Generates text using the language model. |
| `GetBlip2ParameterGradients` | Gets the gradients for all BLIP-2 trainable parameters. |
| `GetBosTokenIdNative` | Resolves the BOS token id from the configured tokenizer (BOS → CLS fallback), so native generation works across OPT/Flan-T5/BERT-style tokenizers rather than assuming a fixed id. |
| `GetCrossAttentionWeights(Tensor<>,String)` | Gets cross-attention weights for visual grounding. |
| `GetEosTokenIdNative` | Resolves the EOS token id from the configured tokenizer (EOS → SEP fallback). |
| `GetImageEmbedding(Tensor<>)` |  |
| `GetImageEmbeddings(IEnumerable<Tensor<>>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetPadTokenIdNative` | Resolves the PAD token id from the configured tokenizer, or null when the tokenizer defines no PAD token — in which case no token is suppressed (suppressing a guessed PAD id could silently drop a valid token). |
| `GetParameters` |  |
| `GetTextEmbedding(String)` |  |
| `GetTextEmbeddingNative(String)` | Gets text embedding using native layers. |
| `GetTextEmbeddingOnnx(String)` | Gets text embedding using ONNX models. |
| `GetTextEmbeddings(IEnumerable<String>)` |  |
| `GroundText(Tensor<>,String)` |  |
| `InitializeLayers` | Initializes layers for both modes. |
| `InitializeNativeLayers(Int32)` | Initializes native mode layers. |
| `InitializeWeights` | Initializes weights with appropriate random values. |
| `PredictCore(Tensor<>)` |  |
| `PrepareImageForOnnx(Tensor<>)` | Prepares image tensor for ONNX inference. |
| `ProjectToLmSpace(Tensor<>)` | Projects Q-Former features to language model space. |
| `ProjectVisionToQformer(Tensor<>)` | Projects vision features to Q-Former dimension. |
| `RetrieveImages(String,IEnumerable<Tensor<>>,Int32,Boolean,Int32)` |  |
| `ScoreCaption(Tensor<>,String)` | Scores a caption against Q-Former features. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetParameters(Vector<>)` |  |
| `SoftmaxScores(Dictionary<String,>)` | Applies softmax normalization to scores. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateImageShape(Tensor<>)` | Validates the shape of an input image tensor. |
| `ZeroShotClassify(Double[],IEnumerable<String>)` |  |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>)` |  |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>,Boolean)` | Performs zero-shot image classification with optional ITM scoring. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDimension` | The dimensionality of the shared embedding space. |
| `_imageSize` | Expected image size (width and height). |
| `_itcProjection` | Image-text contrastive projection. |
| `_itmHead` | Image-text matching head. |
| `_languageModel` | The ONNX inference session for the language model. |
| `_languageModelBackbone` | Type of language model backbone. |
| `_languageModelPath` | Path to the language model ONNX model file. |
| `_languageModelProjection` | Language model projection layer (native mode). |
| `_lmDecoderLayers` | Language model decoder layers for text generation (native mode). |
| `_lmHead` | Language model head for projecting decoder output to vocabulary logits. |
| `_lmHiddenDim` | Language model hidden dimension. |
| `_lossFunction` | Loss function for training. |
| `_maxSequenceLength` | Maximum sequence length for text encoder. |
| `_numHeads` | Number of attention heads in Q-Former. |
| `_numLmDecoderLayers` | Number of LM decoder layers. |
| `_numQformerLayers` | Number of Q-Former layers. |
| `_numQueryTokens` | Number of learnable query tokens. |
| `_optimizer` | Optimizer for training. |
| `_patchEmbedding` | Patch embedding layer for vision. |
| `_patchSize` | Patch size for vision transformer. |
| `_qformer` | The ONNX inference session for the Q-Former. |
| `_qformerCrossAttentionLayers` | Q-Former cross-attention layers (native mode). |
| `_qformerFeedForwardLayers` | Q-Former feed-forward layers (native mode). |
| `_qformerHiddenDim` | Hidden dimension for Q-Former. |
| `_qformerPath` | Path to the Q-Former ONNX model file. |
| `_qformerSelfAttentionLayers` | Q-Former self-attention layers (native mode). |
| `_queryPositionalEmbeddings` | Query positional embeddings. |
| `_queryPositionalEmbeddingsGradients` | Gradient storage for positional embeddings. |
| `_queryTokens` | Learnable query tokens for Q-Former. |
| `_queryTokensGradients` | Gradient storage for query tokens. |
| `_textTokenEmbedding` | Text token embeddings for Q-Former text encoder. |
| `_tokenizer` | The tokenizer for processing text input. |
| `_useNativeMode` | Indicates whether this BLIP-2 network uses native layers (true) or ONNX models (false). |
| `_visionClsToken` | Vision CLS token. |
| `_visionEncoder` | The ONNX inference session for the vision encoder. |
| `_visionEncoderLayers` | Vision transformer layers for image encoding (native mode). |
| `_visionEncoderPath` | Path to the vision encoder ONNX model file. |
| `_visionHiddenDim` | Vision encoder hidden dimension. |
| `_visionPositionalEmbeddings` | Vision positional embeddings. |
| `_vocabularySize` | Vocabulary size for text encoder. |

