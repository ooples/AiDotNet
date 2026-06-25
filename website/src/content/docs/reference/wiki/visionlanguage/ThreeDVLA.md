---
title: "ThreeDVLA<T>"
description: "3D-VLA: connects vision-language-action to 3D world via generative world model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

3D-VLA: connects vision-language-action to 3D world via generative world model.

## For Beginners

3D-VLA is a vision-language-action model that uses 3D world
modeling for robot planning and action generation. Default values follow the original
paper settings.

## How It Works

3D-VLA (UMass, 2024) connects vision-language-action models to the 3D world via a generative
world model. It processes 3D point cloud observations alongside language instructions and
generates both predicted future states (as a world model) and robot actions, enabling
planning through imagination in 3D space.

**References:**

- Paper: "3D-VLA: A 3D Vision-Language-Action Generative World Model (UMass, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from image using 3D-VLA's generative world model approach. |
| `PredictAction(Tensor<>,String)` | Predicts action using 3D-VLA's generative world model approach. |

