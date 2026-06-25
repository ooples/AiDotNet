---
title: "NuSupportVectorClassifier<T>"
description: "Nu-Support Vector Classifier using the nu-SVM formulation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.SVM`

Nu-Support Vector Classifier using the nu-SVM formulation.

## For Beginners

Nu-SVC is an alternative way to control the SVM's complexity:

Standard SVC uses C:

- C controls the penalty for misclassification
- Hard to interpret: "what does C=1.0 mean?"

Nu-SVC uses nu:

- nu is between 0 and 1
- nu is approximately the fraction of support vectors
- nu is also an upper bound on training errors
- More intuitive: "I want about 30% support vectors" means nu=0.3

Use Nu-SVC when:

- You want more interpretable regularization
- You have a target for the error rate
- The C parameter in standard SVC is hard to tune

Note: Nu-SVC and standard SVC produce very similar results when
properly tuned, but nu can be easier to set intuitively.

## How It Works

Nu-SVC uses a different parameterization than standard SVC. Instead of the C parameter,
it uses nu which is an upper bound on the fraction of margin errors and a lower bound
on the fraction of support vectors.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NuSupportVectorClassifier(SVMOptions<>,IRegularization<,Matrix<>,Vector<>>,Double)` | Initializes a new instance of the NuSupportVectorClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeDecisionForSample(Vector<>)` | Computes the decision value for a single sample. |
| `ComputeError(Int32)` | Computes the prediction error for sample i. |
| `ComputeRhoAndIntercept` | Computes rho and intercept. |
| `CreateNewInstance` |  |
| `DecisionFunction(Matrix<>)` |  |
| `Deserialize(Byte[])` |  |
| `ExtractSupportVectors` | Extracts support vectors. |
| `GetModelMetadata` |  |
| `Max(,)` | Returns the maximum of two values. |
| `Min(,)` | Returns the minimum of two values. |
| `Predict(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `TrainNuSMO` | Nu-SMO training algorithm. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alphas` | Alpha coefficients. |
| `_nu` | The nu parameter value. |
| `_random` | Random number generator. |
| `_rho` | Rho parameter (offset in the decision function). |
| `_xTrain` | Stored training features. |
| `_yTrain` | Stored training labels (converted to +1/-1). |

