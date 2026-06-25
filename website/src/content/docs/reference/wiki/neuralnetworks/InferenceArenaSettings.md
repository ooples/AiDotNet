---
title: "InferenceArenaSettings"
description: "Global opt-in switch for the inference forward-caching allocator (#1661 / Tensors #661)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NeuralNetworks`

Global opt-in switch for the inference forward-caching allocator (#1661 / Tensors #661).
When enabled, `Tensor{` runs the forward inside a recycled
`TensorArena` so intermediate-tensor buffers are reused per call (~98% intermediate
allocation reduction, bit-identical output — proven in Tensors #661).

## How It Works

Default `ON` — per the facade pattern, this is an industry-standard zero-config default
(PyTorch's caching allocator is always on for inference): callers get the allocation win without
opting in. Every `NeuralNetworkBase{T}` model routes through the `PredictCore` funnel,
and the forward is bit-identical with the arena on or off. Set `AIDOTNET_INFERENCE_ARENA=0`
to disable (escape hatch), or set `Enabled` in code (e.g. for A/B alloc tests).

Process-wide by design: `TensorArena.Current` is `[ThreadStatic]`, so concurrent
`Predict` on a single model instance already requires external serialization — the same
contract as the existing eval-mode flip in `Tensor{`.

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionDenoiseEnabled` | Whether the multi-step diffusion denoise loop (`DiffusionModelBase.Generate`) opens a per-step `TensorArena`. |
| `Enabled` | Whether `Tensor{` opens a per-call `TensorArena` around the forward. |

