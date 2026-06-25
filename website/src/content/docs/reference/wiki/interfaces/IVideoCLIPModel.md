---
title: "IVideoCLIPModel<T>"
description: "Defines the contract for VideoCLIP-style models that align video and text in a shared embedding space."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for VideoCLIP-style models that align video and text in a shared embedding space.

## For Beginners

VideoCLIP is like CLIP but for videos!

While CLIP matches images with text, VideoCLIP matches VIDEOS with text:

- Understands actions and events that unfold over time
- Can find videos matching text descriptions
- Can generate descriptions for video clips

Key capabilities:

- Temporal understanding: "A person picks up a ball then throws it"
- Action recognition: "Playing basketball", "Cooking", "Dancing"
- Video retrieval: Find videos matching any text query
- Video-text alignment: Match video segments to text descriptions

Architecture differences from CLIP:

- Processes multiple frames, not just one image
- Uses temporal attention/pooling across frames
- Learns motion and action patterns

## How It Works

VideoCLIP extends CLIP's contrastive learning paradigm to the video domain, enabling
text-to-video and video-to-text retrieval, action recognition, and temporal understanding.

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameRate` | Gets the frame rate (frames per second) for video sampling. |
| `NumFrames` | Gets the number of frames the model processes per video clip. |
| `TemporalAggregation` | Gets the temporal aggregation method used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerVideoQuestion(IEnumerable<Tensor<>>,String,Int32)` | Answers a question about video content. |
| `ComputeTemporalSimilarityMatrix(IEnumerable<Tensor<>>,IEnumerable<Tensor<>>)` | Computes temporal similarity matrix between video segments. |
| `ComputeVideoTextSimilarity(String,IEnumerable<Tensor<>>)` | Computes similarity between a text description and a video. |
| `ExtractFrameFeatures(IEnumerable<Tensor<>>)` | Extracts frame-level features before temporal aggregation. |
| `GenerateVideoCaption(IEnumerable<Tensor<>>,Int32)` | Generates a caption describing the video content. |
| `GetVideoEmbedding(IEnumerable<Tensor<>>)` | Converts a video (sequence of frames) into an embedding vector. |
| `GetVideoEmbeddings(IEnumerable<IEnumerable<Tensor<>>>)` | Converts multiple videos into embedding vectors in a batch. |
| `LocalizeMoments(IEnumerable<Tensor<>>,String,Int32)` | Localizes moments in a video that match a text description. |
| `PredictNextAction(IEnumerable<Tensor<>>,IEnumerable<String>)` | Predicts the next action or event in a video. |
| `RetrieveTextsForVideo(IEnumerable<Tensor<>>,IEnumerable<String>,Int32)` | Retrieves the most relevant text descriptions for a video. |
| `RetrieveVideos(String,IEnumerable<Vector<>>,Int32)` | Retrieves the most relevant videos for a text query. |
| `ZeroShotActionRecognition(IEnumerable<Tensor<>>,IEnumerable<String>)` | Performs zero-shot action classification on a video. |

