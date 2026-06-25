---
title: "WeightImporter"
description: "Imports named weight tensors from an `INamedTensorSource` (safetensors or GGUF) into an AiDotNet network's flat parameter vector."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Models.Local`

Imports named weight tensors from an `INamedTensorSource` (safetensors or GGUF) into an
AiDotNet network's flat parameter vector. The caller supplies the tensor names in the network's parameter
order; the importer concatenates their values and calls `SetParameters`.

## For Beginners

Once you know the order a model lays out its weights, hand that list of tensor
names plus a loaded file to this importer and it pours the weights into the model.

## How It Works

AiDotNet networks expose a single flat parameter vector (`GetParameters`/`SetParameters`) rather
than named per-layer tensors, so the architecture-specific part is the *ordering* of source tensor
names that matches that flat layout — supplied by the caller (a per-architecture name list). This class is
the format-agnostic import mechanism; a built-in name map for a concrete architecture is a follow-up that
belongs with that model.

## Methods

| Method | Summary |
|:-----|:--------|
| `Export(NeuralNetworkBase<>)` | Exports the model's parameters as named segments (the inverse of `INamedTensorSource)`), so weights can be saved, inspected, or round-tripped by name. |
| `ImportByName(NeuralNetworkBase<>,INamedTensorSource)` | Imports weights using the model's built-in `ModelParameterMap` name ordering, so the caller does not supply a name list. |
| `ImportInto(NeuralNetworkBase<>,IReadOnlyList<String>,INamedTensorSource)` | Imports the named tensors (in the given order) into the model's parameters. |

