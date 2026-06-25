---
title: "IntervalBoundPropagation<T, TInput, TOutput>"
description: "Implements Interval Bound Propagation (IBP) for certifying neural network robustness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.CertifiedRobustness`

Implements Interval Bound Propagation (IBP) for certifying neural network robustness.

## For Beginners

IBP is like asking "if my input can vary within a certain range,
what is the guaranteed range of possible outputs?" This helps certify that a model's
predictions are stable within a given perturbation radius.

## How It Works

IBP is a formal verification method that propagates interval bounds through neural network
layers to compute guaranteed output bounds for all inputs within a specified perturbation region.
It provides provable robustness guarantees without requiring adversarial examples.

**Mathematical Foundation:**
For a neural network f(x), IBP computes bounds [y_L, y_U] such that:
∀x ∈ B_p(x₀, ε): y_L ≤ f(x) ≤ y_U

For a linear layer with weights W and biases b:

- Input interval: [x_L, x_U]
- W⁺ = max(W, 0), W⁻ = min(W, 0)
- Lower bound: W⁺ · x_L + W⁻ · x_U + b
- Upper bound: W⁺ · x_U + W⁻ · x_L + b

For ReLU activation:

- Lower: max(0, x_L)
- Upper: max(0, x_U)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IntervalBoundPropagation` | Initializes a new instance of the IntervalBoundPropagation class with default options. |
| `IntervalBoundPropagation(CertifiedDefenseOptions<>)` | Initializes a new instance of the IntervalBoundPropagation class with specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySigmoid()` | Applies sigmoid activation to a single value using numerically stable computation. |
| `ApplyTanh()` | Applies tanh activation to a single value using numerically stable computation. |
| `ApproximateBoundsWithSampling(Vector<>,,IFullModel<,,>,)` | Approximates output bounds using sampling when layer access is not available. |
| `CertifyBatch([],IFullModel<,,>)` |  |
| `CertifyPrediction(,IFullModel<,,>)` |  |
| `CheckCertification(Vector<>,Vector<>,Int32)` | Checks if the prediction is certifiably robust. |
| `ComputeCertifiedRadius(,IFullModel<,,>)` |  |
| `ComputeCertifiedRadiusInternal(Vector<>,,IFullModel<,,>,Int32)` | Computes the certified radius using binary search. |
| `ComputeConfidence(Vector<>,Vector<>,Int32)` | Computes the confidence margin between predicted class and runner-up. |
| `ComputeOutputBounds(Vector<>,,IFullModel<,,>,)` | Computes interval bounds for the neural network output. |
| `Deserialize(Byte[])` |  |
| `EvaluateCertifiedAccuracy([],[],IFullModel<,,>,)` |  |
| `GetOptions` |  |
| `GetPredictedClass(Vector<>)` | Gets the predicted class from an output vector. |
| `LoadModel(String)` |  |
| `PropagateActivation(Vector<>,Vector<>,ActivationFunction)` | Propagates interval bounds through an activation function. |
| `PropagateIntervalBounds(Vector<>,Vector<>,List<ILayer<>>)` | Propagates interval bounds through the network layers. |
| `PropagateLeakyReLU(,)` | Applies LeakyReLU activation. |
| `PropagateLinearLayer(Vector<>,Vector<>,Tensor<>,Tensor<>)` | Propagates interval bounds through a linear layer. |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

