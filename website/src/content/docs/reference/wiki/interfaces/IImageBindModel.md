---
title: "IImageBindModel<T>"
description: "Defines the contract for ImageBind models that bind multiple modalities (6+) into a shared embedding space."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for ImageBind models that bind multiple modalities (6+) into a shared embedding space.

## For Beginners

ImageBind connects ALL types of data together!

The breakthrough insight:

- Images are paired with many things: text captions, video audio, depth sensors, etc.
- By learning all these (image, X) pairs, images become a "bridge"
- Now audio and text can be compared, even without audio-text training data!

Six modalities in one model:

1. Images: Regular RGB photos
2. Text: Natural language descriptions
3. Audio: Sound waveforms (speech, music, effects)
4. Video: Moving images (sequences of frames)
5. Thermal: Heat maps from infrared cameras
6. Depth: 3D distance information
7. IMU: Motion sensor data (accelerometer, gyroscope)

Why this matters:

- Search audio by describing sounds in text
- Find images that match a piece of music
- Match thermal images to regular photos
- Universal multimodal understanding!

## How It Works

ImageBind learns a joint embedding space across six modalities: images, text, audio, depth,
thermal, and IMU data. It uses images as a binding modality - since web data contains
many (image, text) pairs, (image, audio) pairs from videos, etc., the model can learn
cross-modal relationships even without direct pairs between all modalities.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimensionality of the shared embedding space. |
| `SupportedModalities` | Gets the list of supported modalities. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAlignment(ModalityType,Object,ModalityType,Object)` | Computes the alignment between two modalities given paired data. |
| `ComputeCrossModalSimilarity(Vector<>,Vector<>)` | Computes similarity between embeddings from any two modalities. |
| `ComputeEmergentAudioTextSimilarity(Tensor<>,String)` | Computes emergent cross-modal relationships without explicit pairing. |
| `CrossModalRetrieval(Vector<>,IEnumerable<Vector<>>,Int32)` | Performs cross-modal retrieval from one modality to another. |
| `FindBestMatch(ModalityType,Object,IEnumerable<ValueTuple<ModalityType,Object>>)` | Finds the best matching modality representation for a query. |
| `FuseModalities(Dictionary<ModalityType,Vector<>>,String)` | Performs multimodal fusion by combining embeddings from multiple modalities. |
| `GenerateDescriptions(ModalityType,Object,IEnumerable<String>,Int32)` | Generates text description for non-text modalities. |
| `GetAudioEmbedding(Tensor<>,Int32)` | Converts audio into a shared embedding vector. |
| `GetDepthEmbedding(Tensor<>)` | Converts depth map into a shared embedding vector. |
| `GetEmbedding(ModalityType,Object)` | Gets embedding for any supported modality using a generic interface. |
| `GetIMUEmbedding(Tensor<>)` | Converts IMU sensor data into a shared embedding vector. |
| `GetImageEmbedding(Tensor<>)` | Converts an image into a shared embedding vector. |
| `GetTextEmbedding(String)` | Converts text into a shared embedding vector. |
| `GetThermalEmbedding(Tensor<>)` | Converts thermal image into a shared embedding vector. |
| `GetVideoEmbedding(IEnumerable<Tensor<>>)` | Converts video into a shared embedding vector. |
| `ZeroShotClassify(ModalityType,Object,IEnumerable<String>)` | Performs zero-shot classification across modalities. |

