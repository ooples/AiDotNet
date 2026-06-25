---
title: "MoCoConfig"
description: "MoCo-specific configuration settings."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SelfSupervisedLearning`

MoCo-specific configuration settings.

## For Beginners

MoCo (Momentum Contrast) uses a memory queue and momentum encoder
to provide consistent negative samples without large batch sizes.

## Properties

| Property | Summary |
|:-----|:--------|
| `Momentum` | Gets or sets the momentum coefficient for the momentum encoder. |
| `QueueSize` | Gets or sets the size of the memory queue. |
| `UseMLPProjector` | Gets or sets whether to use MLP projection head (MoCo v2+). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

