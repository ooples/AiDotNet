---
title: "ControlNetConditionType"
description: "Specifies the type of conditioning input for multi-control ControlNet composition."
section: "API Reference"
---

`Enums` · `AiDotNet.Diffusion.Control`

Specifies the type of conditioning input for multi-control ControlNet composition.

## For Beginners

When using multiple ControlNet conditions at once (e.g., both a sketch
and a depth map), the model needs to know what each input represents so it can combine them
properly. This enum labels each input with its type, like putting labels on different ingredients
before mixing them together.

## How It Works

ControlNet models can accept multiple simultaneous control conditions (e.g., Canny edges + depth map).
This enum identifies the semantic type of each conditioning input to enable proper routing,
weighting, and compositing of multiple control signals.

## Fields

| Field | Summary |
|:-----|:--------|
| `Brightness` | Brightness/luminance map for lighting control. |
| `CannyEdge` | Canny edge detection map for structural guidance. |
| `ColorMap` | Color map for per-pixel color guidance. |
| `ColorPalette` | Color palette for palette-guided generation. |
| `ContentShuffle` | Content shuffle for structure-preserving randomization. |
| `DWPose` | DWPose whole-body keypoint detection. |
| `DepthMap` | Monocular depth estimation for spatial layout. |
| `FaceID` | FaceID embedding for identity-preserving generation. |
| `HED` | HED (Holistically-Nested Edge Detection) soft edges. |
| `InpaintMask` | Binary inpainting mask (white = inpaint region). |
| `InstanceSegmentation` | Instance segmentation with individual object masks. |
| `LineArt` | Line art or clean sketch drawing. |
| `MLSD` | MLSD (Mobile Line Segment Detection) straight lines. |
| `MediaPipeFace` | MediaPipe face mesh landmarks. |
| `NormalMap` | Surface normal map for lighting and geometry. |
| `OpenPose` | OpenPose body and hand keypoints. |
| `QRCode` | QR code pattern for embedding readable codes. |
| `Recolor` | Recoloring target for guided color transfer. |
| `ReferenceImage` | Reference image features (no explicit spatial map). |
| `SAMSegmentation` | SAM (Segment Anything) segmentation mask. |
| `Scribble` | User-drawn scribble or sketch. |
| `SemanticSegmentation` | Semantic segmentation label map. |
| `SoftEdge` | SoftEdge detection (PiDiNet or similar). |
| `StyleReference` | Style reference for style-aligned generation. |
| `Tile` | Tile image for detail-preserving upscaling. |

