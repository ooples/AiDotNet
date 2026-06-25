---
title: "ControlType"
description: "Types of control signals supported by ControlNet."
section: "API Reference"
---

`Enums` · `AiDotNet.Diffusion.Control`

Types of control signals supported by ControlNet.

## Fields

| Field | Summary |
|:-----|:--------|
| `Brightness` | Brightness map control for lighting guidance. |
| `Canny` | Canny edge detection map. |
| `Color` | Color palette control for color guidance. |
| `ColorPalette` | Color palette extraction for palette-guided generation. |
| `ContentShuffle` | Content shuffle for structure-aware style transfer. |
| `DWPose` | DWPose whole-body keypoint detection (improved over OpenPose). |
| `Depth` | Depth map from MiDaS or similar. |
| `FaceID` | FaceID control for identity-preserving face generation. |
| `Hed` | HED (Holistically-Nested Edge Detection). |
| `Inpaint` | Inpaint mask. |
| `InpaintMask` | Inpainting mask for region-specific generation. |
| `LineArt` | Line art/sketch. |
| `MediaPipeFace` | MediaPipe face mesh landmarks for facial control. |
| `Mlsd` | MLSD (Mobile Line Segment Detection). |
| `Normal` | Surface normal map. |
| `Pose` | OpenPose body keypoints. |
| `QR` | QR code pattern control for embedding QR codes in images. |
| `Recolor` | Recolor control for guided recoloring of images. |
| `Reference` | Reference-only control using image features without explicit conditioning. |
| `SAMSegment` | SAM (Segment Anything Model) segmentation maps. |
| `Scribble` | User-drawn scribbles. |
| `Segmentation` | Semantic segmentation map. |
| `Shuffle` | Shuffle/random structure. |
| `SoftEdge` | SoftEdge detection. |
| `Tile` | Tile upscaling control for detail preservation. |

