---
title: "NeuronAttributionExplainer<T>"
description: "Neuron-level attribution explainer for understanding individual neuron contributions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Neuron-level attribution explainer for understanding individual neuron contributions.

## For Beginners

While most attribution methods explain which INPUT FEATURES matter,
neuron attribution explains which NEURONS IN A LAYER contribute to the output.

**Why is this useful?**

- **Understanding hidden representations:** What did the model learn in each layer?
- **Feature discovery:** Which neurons encode which concepts?
- **Debugging:** Are certain neurons always/never active?
- **Pruning:** Which neurons can be removed without hurting performance?

**Supported methods:**

- **NeuronGradient:** Simple gradient of output w.r.t. neuron activation
- **NeuronIntegratedGradients:** Integrated Gradients from baseline to actual activation
- **NeuronConductance:** Combines gradient and activation (like Input×Gradient for neurons)

**Example use case:**
In a CNN for image classification, you might find that neuron #42 in the last conv layer
has high attribution for "cat" predictions. Investigating what activates neuron #42 could
reveal it's a "whisker detector".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuronAttributionExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Int32,>,Int32,NeuronAttributionMethod,Int32,String[])` | Initializes a neuron attribution explainer. |

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
| `ComputeConductanceAttribution(Vector<>,Int32,Vector<>)` | Computes conductance attribution for neurons. |
| `ComputeGradientAttribution(Vector<>,Int32)` | Computes simple gradient attribution for neurons. |
| `ComputeIntegratedGradientsAttribution(Vector<>,Int32,Vector<>)` | Computes Integrated Gradients attribution for neurons. |
| `ComputeNeuronAttribution(Vector<>,Nullable<Int32>)` | Computes neuron attribution for all neurons in the layer. |
| `GetPredictedClass(Vector<>)` | Gets the predicted class from output. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

