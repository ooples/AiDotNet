---
title: "SAM2<T>"
description: "Segment Anything Model 2 (SAM2) for video object segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Segmentation`

Segment Anything Model 2 (SAM2) for video object segmentation.

## For Beginners

SAM2 is a powerful model that can segment any object in video.
You can interact with it by:

- Clicking on an object in the first frame to select it
- Drawing a bounding box around objects
- Providing text prompts describing what to segment

Once you identify an object, SAM2 automatically tracks and segments it across
all frames in the video, even when the object moves, rotates, or is partially occluded.

Common use cases:

- Video editing (isolating subjects for effects)
- Object tracking and analysis
- Video annotation and labeling
- Interactive video manipulation

## How It Works

**Technical Details:**

- Memory attention mechanism for temporal consistency
- Hierarchical image encoder (similar to MAE/ViT)
- Prompt encoder for points, boxes, and masks
- Mask decoder with occlusion prediction
- Memory bank for efficient object tracking

**Reference:** Ravi et al., "SAM 2: Segment Anything in Images and Videos"
Meta AI, 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAM2` | Initializes a new instance of the SAM2 class in native (trainable) mode. |
| `SAM2(NeuralNetworkArchitecture<>,String,SAM2ModelSize,Int32,SAM2Options)` | Initializes a new instance of the SAM2 class in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentMemorySize` | Gets the current memory bank size. |
| `InputHeight` | Gets the input height. |
| `InputWidth` | Gets the input width. |
| `ModelSize` | Gets the model size variant. |
| `SupportsTraining` | Gets whether training is supported. |
| `UseNativeMode` | Gets whether using native mode (trainable) or ONNX mode (inference only). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearMemory` | Clears the memory bank for starting a new video. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EncodeImageOnnx(Tensor<>)` | Encodes an image using the ONNX model. |
| `GetModelMetadata` |  |
| `GetOcclusionScore(Tensor<>,Single[0:,0:],Int32[])` | Gets the occlusion score for the current segmentation. |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `InteractiveVideoSegmentation(List<Tensor<>>,Dictionary<Int32,ValueTuple<Single[0:,0:],Int32[]>>)` | Performs interactive video segmentation with refinement. |
| `PredictCore(Tensor<>)` |  |
| `SegmentWithBox(Tensor<>,Single[])` | Segments objects in an image given a bounding box. |
| `SegmentWithMask(Tensor<>,Tensor<>)` | Segments objects using a mask prompt (for refinement). |
| `SegmentWithPoints(Tensor<>,Single[0:,0:],Int32[])` | Segments objects in an image given point prompts. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `TrackObject(List<Tensor<>>,Single[0:,0:],Int32[])` | Tracks and segments an object across video frames. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

