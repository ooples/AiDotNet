---
title: "CROWNVerification<T, TInput, TOutput>"
description: "Implements CROWN (Convex Relaxation based perturbation analysis Of Neural networks) for computing certified robustness bounds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.CertifiedRobustness`

Implements CROWN (Convex Relaxation based perturbation analysis Of Neural networks)
for computing certified robustness bounds.

## For Beginners

CROWN finds the smallest "safety box" around a prediction
where we can guarantee the model's answer won't change. It's more precise than IBP
because it uses smarter mathematical approximations.

## How It Works

CROWN is a state-of-the-art neural network verification technique that computes
tighter certified bounds than IBP by using linear relaxation of non-linear activations.
It works by propagating linear upper and lower bounds backward through the network.

**Mathematical Foundation:**
For a ReLU activation σ(x) = max(0, x) with bounds [l, u]:

Case 1: l ≥ 0 (always active): σ(x) = x
Case 2: u ≤ 0 (always inactive): σ(x) = 0
Case 3: l < 0 < u (crossing): Use linear relaxation

- Upper bound: σ(x) ≤ u(x - l)/(u - l)
- Lower bound: σ(x) ≥ 0 or σ(x) ≥ x (choose tighter)

CROWN propagates these linear bounds backward through the network to get
tighter output bounds than IBP's forward propagation alone.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CROWNVerification` | Initializes a new instance of the CROWNVerification class with default options. |
| `CROWNVerification(CertifiedDefenseOptions<>)` | Initializes a new instance of the CROWNVerification class with specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CertifyBatch([],IFullModel<,,>)` |  |
| `CertifyPrediction(,IFullModel<,,>)` |  |
| `ComputeCROWNBounds(Vector<>,,IFullModel<,,>,)` | Computes CROWN bounds for the neural network output. |
| `ComputeCROWNBoundsWithLayers(Vector<>,,List<ILayer<>>,Vector<>,Vector<>)` | Computes CROWN bounds using backward linear bound propagation. |
| `ComputeCertifiedRadius(,IFullModel<,,>)` |  |
| `ComputeIBPBounds(Vector<>,,IFullModel<,,>,)` | Computes forward IBP bounds for pre-activation intervals. |
| `ComputeReLUCROWNBounds(,)` | Computes the CROWN linear relaxation bounds for ReLU. |
| `Deserialize(Byte[])` |  |
| `EvaluateCertifiedAccuracy([],[],IFullModel<,,>,)` |  |
| `GetOptions` |  |
| `LoadModel(String)` |  |
| `PropagateActivationCROWN(Vector<>,Vector<>,,,Vector<>,Vector<>,ActivationFunction)` | Propagates linear bounds backward through an activation function using CROWN relaxation. |
| `PropagateActivationIBP(Vector<>,Vector<>,ActivationFunction)` | Propagates interval bounds through an activation using IBP. |
| `PropagateIBPBounds(Vector<>,Vector<>,List<ILayer<>>)` | Propagates IBP bounds through layers. |
| `PropagateLinearCROWN(Vector<>,Vector<>,,,Tensor<>,Tensor<>)` | Propagates linear bounds backward through a linear layer. |
| `PropagateLinearLayer(Vector<>,Vector<>,Tensor<>,Tensor<>)` | Propagates interval bounds through a linear layer. |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

