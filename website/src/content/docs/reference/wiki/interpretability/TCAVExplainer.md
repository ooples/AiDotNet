---
title: "TCAVExplainer<T>"
description: "Testing with Concept Activation Vectors (TCAV) explainer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Testing with Concept Activation Vectors (TCAV) explainer.

## For Beginners

TCAV is a technique that explains model predictions using
human-understandable concepts instead of raw features. While most explanation methods
tell you which pixels or features matter, TCAV tells you which CONCEPTS matter.

**Example:** Instead of highlighting pixels that matter for a "doctor" prediction,
TCAV can tell you whether the concept of "stethoscope" or "white coat" influenced
the prediction.

**How TCAV works:**

1. **Collect concept examples:** Gather images/examples that represent your concept

(e.g., images with stripes for a "striped" concept)

2. **Collect random examples:** Gather examples that don't specifically represent the concept
3. **Train a CAV:** Train a linear classifier to distinguish concept vs random at a

specific layer. The classifier's weight vector becomes the Concept Activation Vector (CAV).

4. **Compute directional derivatives:** See how sensitive the model output is to moving

in the direction of the CAV

5. **Compute TCAV score:** The fraction of test inputs where moving toward the concept

increases the model's prediction for a class

**Interpreting TCAV scores:**

- TCAV score = 0.8 means 80% of images are more likely to be classified as the target

class when they have more of the concept

- TCAV score = 0.5 means the concept doesn't influence the prediction (random)
- TCAV score = 0.2 means the concept DECREASES the likelihood of the target class

**Statistical significance:** TCAV runs multiple times with different random samples
to ensure the score is statistically significant and not due to random noise.

**When to use TCAV:**

- When you want high-level, human-understandable explanations
- When you have examples of concepts you want to test
- For testing fairness/bias (e.g., does "gender" concept affect hiring predictions?)
- For debugging models (e.g., is my model relying on spurious correlations?)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TCAVExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Int32,Double,Int32,Double,Nullable<Int32>)` | Initializes a new TCAV explainer. |

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
| `ComputeDirectionalDerivative(Vector<>,ConceptActivationVector<>,Int32)` | Computes the directional derivative of the model output with respect to the CAV. |
| `ComputeLayerGradientFiniteDiff(INeuralNetwork<>,Vector<>,Int32,Func<Vector<>,Vector<>>,Int32)` | Computes layer gradients using finite differences. |
| `ComputeTCAVScore(Matrix<>,ConceptActivationVector<>,Int32)` | Computes the TCAV score for a concept on a set of inputs. |
| `ExplainLocal(Vector<>,ConceptActivationVector<>,Int32,String)` | Computes concept sensitivity for a single input. |
| `FromNetwork(INeuralNetwork<>,Func<Vector<>,Vector<>>,Int32,Int32,Double,Nullable<Int32>)` | Creates a TCAV explainer from a neural network using input gradient helper. |
| `NormalCDF(Double)` | Standard normal CDF approximation. |
| `RunTCAV(Matrix<>,Matrix<>,Matrix<>,Int32,String)` | Runs a full TCAV analysis with statistical significance testing. |
| `RunTCAVMultiple(Dictionary<String,Matrix<>>,Matrix<>,Matrix<>,Int32)` | Runs TCAV analysis for multiple concepts. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `SolvePositiveDefinite(Matrix<>,Vector<>)` | Solves a positive definite system using Cholesky decomposition. |
| `TrainCAV(Matrix<>,Matrix<>)` | Trains a Concept Activation Vector from concept and random examples. |
| `TrainLinearClassifier(Matrix<>,Vector<>)` | Trains a linear classifier using regularized least squares. |

