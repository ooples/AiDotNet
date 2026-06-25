---
title: "SegmentationBuilderExtensions"
description: "Extension methods for image segmentation operations through the AiModelBuilder facade."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet`

Extension methods for image segmentation operations through the AiModelBuilder facade.

## For Beginners

These methods let you perform segmentation on images after configuring
a model with `IFullModel{`. Each method
detects what kind of segmentation model you configured and calls the appropriate interface.

Usage pattern:

## Methods

| Method | Summary |
|:-----|:--------|
| `CorrectSegmentationMask(AiModelBuilder<,Tensor<>,Tensor<>>,Int32,Tensor<>)` | Corrects a tracked object's mask at the current frame. |
| `GetSegmentationClassCount(AiModelBuilder<,Tensor<>,Tensor<>>)` | Gets the number of segmentation classes the configured model predicts. |
| `GetSegmentationInputSize(AiModelBuilder<,Tensor<>,Tensor<>>)` | Gets the expected input dimensions of the configured segmentation model. |
| `GetSemanticClassMap(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Gets the per-pixel class map (argmax of logits) from a semantic segmentation model. |
| `GetSemanticProbabilities(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Gets per-pixel class probability maps from a semantic segmentation model. |
| `InitializeVideoTracking(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,Tensor<>,Int32[])` | Initializes video object tracking with first-frame masks. |
| `IsSegmentationOnnxMode(AiModelBuilder<,Tensor<>,Tensor<>>)` | Gets whether the configured segmentation model is running in ONNX mode. |
| `PropagateSegmentation(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Propagates tracked object masks to the next video frame. |
| `ResetSegmentationTracking(AiModelBuilder<,Tensor<>,Tensor<>>)` | Resets the video tracking state and memory. |
| `SegmentEverything(AiModelBuilder<,Tensor<>,Tensor<>>)` | Automatically segments everything in the image without prompts. |
| `SegmentFromBox(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Segments the object inside a bounding box. |
| `SegmentFromConversation(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,IReadOnlyList<ValueTuple<String,String>>,String)` | Segments objects from a multi-turn conversational context. |
| `SegmentFromExpression(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,String)` | Segments objects described by a natural language expression with reasoning. |
| `SegmentFromMask(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Refines a rough mask into a precise segmentation. |
| `SegmentFromPoints(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,Tensor<>)` | Segments the region indicated by point prompts (click to segment). |
| `SegmentInstances(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Detects and segments individual object instances in an image. |
| `SegmentMedicalFewShot(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,Tensor<>,Tensor<>)` | Segments using few-shot examples (for models like UniverSeg, MedSAM). |
| `SegmentMedicalSlice(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Segments a 2D medical image slice (CT, MRI, X-ray, etc.). |
| `SegmentMedicalVolume(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Segments a 3D medical volume (full CT or MRI scan). |
| `SegmentPanoptic(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Performs panoptic segmentation (unified semantic + instance) on an image. |
| `SegmentSemantic(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Produces raw per-pixel class logits from a semantic segmentation model. |
| `SegmentVideoFromExpression(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,String)` | Segments and tracks objects in a video from a natural language description. |
| `SegmentWithTextClasses(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,IReadOnlyList<String>)` | Segments objects described by text class names (open-vocabulary). |
| `SegmentWithTextPrompt(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,String)` | Segments objects matching a single text prompt (grounded segmentation). |
| `SetSegmentationImage(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Encodes an image for subsequent prompt-based segmentation. |

