---
title: "OcclusionExplainer<T>"
description: "Occlusion explainer for image and sequential data interpretation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Occlusion explainer for image and sequential data interpretation.

## For Beginners

Occlusion is a simple but powerful explanation technique.
The idea is: systematically hide different parts of the input and see how the
prediction changes.

**How it works:**

1. Take your input (e.g., an image)
2. Place a "patch" over one part of the input (occlude it)
3. See how the model's prediction changes
4. Move the patch and repeat
5. The result is a map showing which regions matter most

**Intuition:** If covering a region causes the prediction to drop significantly,
that region was important for the prediction.

**Use cases:**

- Understanding which parts of an image a classifier looks at
- Finding what regions of a medical scan led to a diagnosis
- Debugging models that use spurious correlations (e.g., looking at background)

**Advantages:**

- Very simple and intuitive
- Model-agnostic (works with any model)
- Easy to visualize

**Disadvantages:**

- Can be slow (many forward passes required)
- May not capture feature interactions well
- Results depend on occlusion patch size

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OcclusionExplainer(Func<Tensor<>,Tensor<>>,Int32[],Int32[],,OcclusionShape)` | Initializes a new Occlusion explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` | Gets the method name. |
| `SupportsGlobalExplanations` | Gets whether this explainer supports global explanations. |
| `SupportsLocalExplanations` | Gets whether this explainer supports local explanations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCircularOcclusion(Tensor<>,Int32[])` | Applies circular occlusion. |
| `ApplyOcclusion(Tensor<>,Int32[])` | Applies occlusion at the given position. |
| `ApplyRectangularOcclusion(Tensor<>,Int32[])` | Applies rectangular occlusion. |
| `ComputeOutputShape(Int32[])` | Computes the output shape for the sensitivity map. |
| `Explain(Tensor<>,Nullable<Int32>)` | Explains a single input by computing occlusion sensitivity. |
| `Explain(Vector<>)` | Explains a single vector input. |
| `ExplainBatch(List<Tensor<>>,Nullable<Int32>)` | Explains a batch of inputs. |
| `ExplainBatch(Matrix<>)` | Explains a batch of inputs. |
| `ForImages(Func<Tensor<>,Tensor<>>,Int32,Int32,Nullable<Int32>,Nullable<Int32>,,OcclusionShape)` | Initializes an Occlusion explainer for 2D images. |
| `GenerateOcclusionPositions(Int32[])` | Generates all occlusion positions. |
| `GetPredictedClass(Tensor<>)` | Gets the predicted class from output. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `SetSensitivityValue(Tensor<>,Int32,)` | Sets a sensitivity value in the map. |

