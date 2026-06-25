---
title: "DeserializationHelper"
description: "DeserializationHelper — Helpers & Utilities in AiDotNet.Helpers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

_No summary documentation available yet._

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildPlaceholderHeterogeneousGraphMetadata(Type)` | Builds a minimal valid `HeterogeneousGraphMetadata` for layer reconstruction: one node type ("default"), one self-loop edge type ("default" → "default"), 64-dim node features. |
| `ConstructLoRAAdapterWithValidation(Type,Type,Dictionary<String,Object>,String)` | Construct a LoRA adapter with constraint-aware defaults. |
| `CreateGRULayer(Type,Int32[],Int32[],Dictionary<String,Object>)` | Creates a GRU layer during deserialization. |
| `CreateLSTMLayer(Type,Int32[],Int32[],Dictionary<String,Object>)` | Creates an LSTM layer during deserialization. |
| `CreateLayerFromType(String,Int32[],Int32[],Dictionary<String,Object>)` | Creates a layer of the specified type during deserialization. |
| `DeserializeInterface(BinaryReader)` | Deserializes and creates an instance of an interface based on the type name read from a BinaryReader. |
| `EnsureLoRASharedMatricesInitialized(Type)` | Calls the adapter type's `InitializeSharedMatrices(int, int, int)` static method with sensible defaults so reflection-based reconstruction can construct an adapter instance afterwards. |
| `IsLoRAAdapterRequiringSharedMatrices(Type)` | True when the adapter type requires `InitializeSharedMatrices` to be called once before any instance is constructed. |
| `IsLoRAAdapterWithSpecificValidation(Type)` | True when this LoRA adapter's constructor performs validation that the generic matcher's HP defaults can't satisfy without targeted tweaks. |
| `IsMissingCtorMessage(String)` | Defensive fallback: returns true when an InvalidOperationException's message matches the legacy "Cannot find <layer name> constructor" convention. |
| `TryConstructByMatchingMetadata(Type,Int32[],Int32[],Dictionary<String,Object>,String)` | Reflection-driven constructor matcher used as the universal fallback when no dedicated branch exists for a layer. |
| `TryConstructInnerLayerFromMetadata(Dictionary<String,Object>)` | Reads InnerLayerTypeName / InnerLayerInputShape / InnerLayerOutputShape from `additionalParams` (written by `GetMetadata`) and recursively builds the wrapped layer via `Object})`. |
| `TryCreatePlaceholderInnerLayer(Type,Object)` | Builds a placeholder inner-layer instance when a constructor parameter expects an `ILayer<T>` / `LayerBase<T>` / `ILayer<T>[]` / `List<ILayer<T>>` / `IEnumerable<ILayer<T>>` reference. |
| `TryDefaultMlDoubleHyperparameter(String)` | Sensible-default lookup for double-typed ML hyperparameters — analog of `String)`. |
| `TryDefaultMlIntHyperparameter(String)` | Sensible-default lookup for int-typed ML hyperparameters whose names don't match the input/output-shape naming heuristic. |
| `TryRestoreActivation(Dictionary<String,Object>)` | Tries to restore an activation function from serialized metadata. |

