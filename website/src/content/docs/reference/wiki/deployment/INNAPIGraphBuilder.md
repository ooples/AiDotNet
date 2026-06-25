---
title: "INNAPIGraphBuilder"
description: "Translates a model file (TFLite / ONNX / etc.) into an NNAPI operation graph by adding operands and operations to a freshly-created `ANeuralNetworksModel` handle."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Deployment.Mobile.Android`

Translates a model file (TFLite / ONNX / etc.) into an NNAPI operation
graph by adding operands and operations to a freshly-created
`ANeuralNetworksModel` handle. Plug an implementation into
`NNAPIBackend` via its constructor so `LoadModel`
can actually compile a real NNAPI graph instead of falling back to the
managed CPU executor.

## How It Works

Each NNAPI backend instance owns its own `_model` IntPtr. The
builder is invoked once per `String)`
call, after the backend has called `ANeuralNetworksModel_create`
but before `ANeuralNetworksModel_finish`. The builder is
responsible for:

Splitting graph construction out of the backend keeps the backend
model-format-agnostic — TFLite, ONNX, and bespoke IRs all plug in via
the same interface.

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildGraph(IntPtr,Byte[])` | Populates the supplied `modelHandle` with operands and operations decoded from `modelBytes`. |

