---
title: "Octo<T>"
description: "Octo: open-source generalist robot policy trained on 800K demonstrations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

Octo: open-source generalist robot policy trained on 800K demonstrations.

## For Beginners

Octo is an open-source vision-language-action model for generalist
robot control across different robot platforms. Default values follow the original paper
settings.

## How It Works

Octo (Berkeley, 2024) is an open-source generalist robot policy trained on 800K demonstrations
from the Open X-Embodiment dataset. It uses a transformer backbone that processes interleaved
image observations and language task specifications, outputting continuous robot actions
that generalize across different robot embodiments and manipulation tasks.

**References:**

- Paper: "Octo: An Open-Source Generalist Robot Policy (Berkeley, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from image using Octo's readout token + task-conditioned attention. |
| `PredictAction(Tensor<>,String)` | Predicts action using Octo's readout token + diffusion head architecture. |

