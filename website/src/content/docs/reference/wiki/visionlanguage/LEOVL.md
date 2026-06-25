---
title: "LEOVL<T>"
description: "LEO-VL: efficient 3D scene representation from multi-view RGB-D."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.ThreeD`

LEO-VL: efficient 3D scene representation from multi-view RGB-D.

## For Beginners

LEO-VL is a vision-language model for 3D scene understanding
using multi-view RGB-D camera inputs. Default values follow the original paper settings.

## How It Works

LEO-VL (2025) provides efficient 3D scene understanding from multi-view RGB-D inputs.
It constructs 3D scene representations from multiple depth-camera views without requiring
explicit 3D reconstruction, using a view aggregation module to fuse multi-view features
into a coherent spatial representation for language-guided scene understanding and navigation.

**References:**

- Paper: "LEO-VL: Efficient 3D Scene Understanding via Multi-View RGB-D (Various, 2025)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFrom3D(Tensor<>,String)` | Processes 3D point cloud using LEO-VL's embodied spatial reasoning approach. |
| `GenerateFromImage(Tensor<>,String)` | Generates from 2D image using LEO-VL's embodied multi-modal approach. |

