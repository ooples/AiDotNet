---
title: "InfluenceFunctionExplainer<T>"
description: "Influence Function explainer for training data attribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Influence Function explainer for training data attribution.

## For Beginners

Influence Functions answer the question: "Which training examples
were most responsible for this prediction?"

This is different from feature attribution (which features matter) - instead it tells
you which TRAINING DATA points matter. This is incredibly useful for:

**Use Cases:**

- **Debugging:** Finding mislabeled training data
- **Data cleaning:** Identifying harmful training examples
- **Understanding:** Seeing which examples the model learned from
- **Fairness:** Finding training data that causes biased predictions

**How it works:**
Influence Functions use calculus to efficiently approximate: "What would happen to
this test prediction if we removed a specific training example and retrained?"

Instead of actually retraining (which is expensive), we use the Hessian (second
derivatives of the loss) to estimate the effect mathematically.

**The math (simplified):**
influence(training_point) = (gradient_test) * (inverse_Hessian) * (gradient_train)

- gradient_test: How the test loss changes with parameters
- inverse_Hessian: How parameter changes propagate through the model
- gradient_train: How this training point affected the parameters

**Interpretation:**

- Positive influence: Removing this training point would HURT test performance
- Negative influence: Removing this training point would HELP test performance
- Large magnitude: This training point had a big effect

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InfluenceFunctionExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>,>,Func<Vector<>,Vector<>,Vector<>>,Matrix<>,Vector<>,InverseHessianMethod,Double,Int32,Int32,Double,Nullable<Int32>,Func<Vector<>,Vector<>,Vector<>,Vector<>>)` | Initializes a new Influence Function explainer with custom gradient function. |
| `InfluenceFunctionExplainer(INeuralNetwork<>,Func<Vector<>,Vector<>,>,Matrix<>,Vector<>,InverseHessianMethod,Double,Int32,Int32,Double,Nullable<Int32>,Func<Vector<>,Vector<>,Vector<>,Vector<>>)` | Initializes a new Influence Function explainer. |

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
| `ComputeAverageGradient` | Computes average gradient over training data. |
| `ComputeAverageHessianVectorProduct(Vector<>)` | Computes average Hessian-vector product over training data. |
| `ComputeGradient(Vector<>,Vector<>)` | Computes the gradient of the loss for a single sample w.r.t. |
| `ComputeHessianVectorProduct(Vector<>,Vector<>,Vector<>)` | Computes Hessian-vector product for a single sample. |
| `ComputeIHVP_ConjugateGradient(Vector<>)` | Computes inverse Hessian-vector product using conjugate gradient. |
| `ComputeIHVP_Direct(Vector<>)` | Computes inverse Hessian-vector product directly (only for small models). |
| `ComputeIHVP_LiSSA(Vector<>)` | Computes inverse Hessian-vector product using LiSSA. |
| `ComputeInfluence(Vector<>,)` | Computes the influence of all training samples on a test sample. |
| `ComputeInverseHessianVectorProduct(Vector<>)` | Computes inverse Hessian-vector product using the selected method. |
| `ComputeSelfInfluence` | Computes the self-influence of each training sample. |
| `ComputeTracIn(Vector<>,,List<Matrix<>>)` | Computes TracIn-style influence using gradient checkpoints. |
| `EnsureHvpAvailable` | Throws `NotSupportedException` with clear guidance if the caller did not supply an `hvpFunction` at construction time. |
| `EnsureTrainingGradientsComputed` | Ensures training gradients are computed and cached. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `SolvePositiveDefinite(Matrix<>,Vector<>)` | Solves a positive definite system using Cholesky decomposition. |
| `ValidateHyperparameters(Double,Int32,Int32,Double)` | Asserts that all numeric hyperparameters are in their valid ranges. |
| `ValidateTrainingShapes(Matrix<>,Vector<>)` | Asserts that the training matrix has at least one row and that the label vector length matches `Rows`. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hvpFunction` | Optional caller-supplied parameter-space Hessian-vector product `(input, target, vector) -> H_θ * vector`. |

