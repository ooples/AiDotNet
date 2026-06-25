---
title: "BYOLConfig"
description: "BYOL-specific configuration settings."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SelfSupervisedLearning`

BYOL-specific configuration settings.

## For Beginners

BYOL (Bootstrap Your Own Latent) learns without negative samples
by using an asymmetric architecture with a predictor network.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseMomentum` | Gets or sets the base momentum for the target encoder. |
| `FinalMomentum` | Gets or sets the final momentum (for momentum scheduling). |
| `PredictorHiddenDimension` | Gets or sets the hidden dimension of the predictor MLP. |
| `PredictorOutputDimension` | Gets or sets the output dimension of the predictor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

