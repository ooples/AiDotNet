---
title: "VideoCLIP<T>"
description: "VideoCLIP model for video-text understanding and retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Understanding`

VideoCLIP model for video-text understanding and retrieval.

## For Beginners

VideoCLIP learns to understand both videos and text descriptions
in a shared "embedding space" where similar concepts are close together.

Key capabilities:

- Video-to-Text Search: Find text descriptions that match a video
- Text-to-Video Search: Find videos that match a text query
- Zero-Shot Classification: Classify videos into categories without training
- Video Captioning: Generate descriptions for videos
- Video Question Answering: Answer questions about video content

The model creates embeddings (numerical representations) for both videos and text
that can be compared using similarity measures. Videos and their corresponding
descriptions will have similar embeddings.

## How It Works

**Technical Details:**

- Contrastive learning on video-text pairs
- Temporal transformer for video understanding
- Text transformer for language understanding
- Joint embedding space with cosine similarity
- Pre-trained on large-scale video-text datasets

**Reference:** Xu et al., "VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding"
EMNLP 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoCLIP` | Initializes a new instance with default architecture settings. |
| `VideoCLIP(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double,String,String,VideoCLIPVideoOptions)` | Initializes a new instance of the VideoCLIP class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `InputHeight` | Gets the video frame height. |
| `InputWidth` | Gets the video frame width. |
| `NumFrames` | Gets the number of frames processed. |
| `SupportsTraining` | Gets whether training is supported. |
| `Temperature` | Gets or sets the temperature parameter for softmax. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Element-wise tensor addition for residual connections. |
| `ApplyQuickGELU(Tensor<>)` | Quick GELU approximation as used in OpenAI CLIP. |
| `ComputeSimilarity(Tensor<>,Tensor<>)` | Computes similarity between video and text embeddings. |
| `ComputeSimilarityMatrix(List<Tensor<>>,List<Tensor<>>)` | Computes video-text similarity matrix for a batch. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EncodeText(Tensor<>)` | Encodes text into an embedding vector. |
| `EncodeVideo(Tensor<>)` | Encodes a video into an embedding vector. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeEmbeddingTable(Tensor<>,Int32,Int32)` | Initializes an embedding table with Xavier/Glorot uniform initialization. |
| `InitializeLayers` |  |
| `LookupTokenEmbeddings(Tensor<>)` | Performs embedding lookup from the token embedding table. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `TextFFN(Tensor<>,Int32)` | Text transformer feed-forward network with quick GELU activation. |
| `TextLayerNorm(Tensor<>)` | Layer normalization for text transformer. |
| `TextMultiHeadAttention(Tensor<>,Int32)` | Text transformer multi-head self-attention following CLIP architecture. |
| `TextToVideoRetrieval(String,List<Tensor<>>,Int32)` | Retrieves the most similar videos to a text query. |
| `Tokenize(String)` | Tokenizes text using the CLIP BPE tokenizer. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `VideoToTextRetrieval(Tensor<>,List<String>,Int32)` | Retrieves the most similar text descriptions for a video. |
| `ZeroShotClassify(Tensor<>,List<String>)` | Performs zero-shot video classification. |

