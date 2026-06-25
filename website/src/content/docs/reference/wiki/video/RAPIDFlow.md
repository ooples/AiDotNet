---
title: "RAPIDFlow<T>"
description: "RAPIDFlow — Recurrent Adaptable Pyramids with Iterative Decoding for efficient optical-flow estimation (Morimitsu 2025, ``)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

RAPIDFlow — Recurrent Adaptable Pyramids with Iterative Decoding for
efficient optical-flow estimation (Morimitsu 2025,
``).

## For Beginners

Imagine looking at two video frames side by side
and trying to figure out how things moved between them. RAPIDFlow zooms
OUT first (the pyramid downsample), figures out the big motions in the
blurry low-resolution view (the iterative refinement at the bottom of
the pyramid), then progressively zooms IN to sharpen those estimates back
to full resolution (the upsample decoder). Working at lower resolutions
is what makes it fast — there are far fewer pixels to process at the
bottom of the pyramid.

## How It Works

Paper-faithful architecture summary:

The previous implementation in this codebase was a flat stack of 10
non-pyramid full-resolution convolutions — that violated both the
"Pyramid" axis of the paper (no multi-scale structure) and the
"Efficient" axis (every conv at full res = ~15× more compute than the
paper-faithful pyramid). Rewriting to the pyramid structure here is
what fixes the generated `Training_ShouldReduceLoss` /
`MoreData_ShouldNotDegrade` /
`LossStrictlyDecreasesOnMemorizationTask` invariants — the
failure mode was real (the model couldn't be exercised within the
xUnit per-test timeout, not the test budget being too tight).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RAPIDFlow` | Initializes a new RAPIDFlow with paper-default settings (256×256 RGB inputs, 5 refinement iterations). |
| `RAPIDFlow(NeuralNetworkArchitecture<>,Int32,RAPIDFlowOptions)` | Creates a new RAPIDFlow model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EstimateFlow(Tensor<>,Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `TryGetArchitectureInputShape` | RAPIDFlow consumes two RGB frames concatenated channel-wise — 2 × Architecture.InputDepth = 6 channels at the first encoder Conv — but Architecture.InputDepth itself reports the SINGLE-FRAME count (3) so it matches the architecture's per-fr… |
| `UpdateParameters(Vector<>)` |  |

