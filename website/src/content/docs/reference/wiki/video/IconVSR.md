---
title: "IconVSR<T>"
description: "IconVSR: information-aggregation with coupled propagation for video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

IconVSR: information-aggregation with coupled propagation for video super-resolution.

## For Beginners

In long videos, processing frames one-by-one can accumulate errors.
IconVSR solves this by picking "keyframes" as reliable reference points and letting the
forward/backward passes share information, like having checkpoints in a relay race.

**Usage:**

## How It Works

IconVSR (Chan et al., CVPR 2021) extends BasicVSR with two key innovations:

- Information-Aggregation Module: selects sparsely-distributed keyframes and uses their

high-quality features as anchors during recurrent propagation, preventing error
accumulation in long sequences

- Coupled Propagation: forward and backward propagation branches share intermediate

features through a coupling mechanism, improving information flow

This improves temporal consistency for long video sequences compared to BasicVSR.

**Reference:** "BasicVSR: The Search for Essential Components in Video Super-Resolution
and Beyond" (Chan et al., CVPR 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IconVSR(NeuralNetworkArchitecture<>,IconVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an IconVSR model in native training mode. |
| `IconVSR(NeuralNetworkArchitecture<>,String,IconVSROptions)` | Creates an IconVSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

