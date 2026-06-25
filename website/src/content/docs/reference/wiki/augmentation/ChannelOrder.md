---
title: "ChannelOrder"
description: "Specifies the channel ordering of an image tensor."
section: "API Reference"
---

`Enums` · `AiDotNet.Augmentation.Image`

Specifies the channel ordering of an image tensor.

## Fields

| Field | Summary |
|:-----|:--------|
| `BCHW` | Batch, Channels, Height, Width (batched CHW). |
| `BHWC` | Batch, Height, Width, Channels (batched HWC). |
| `CHW` | Channels, Height, Width (PyTorch/Caffe convention). |
| `HWC` | Height, Width, Channels (TensorFlow/NumPy convention). |

