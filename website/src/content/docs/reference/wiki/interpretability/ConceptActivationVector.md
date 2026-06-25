---
title: "ConceptActivationVector<T>"
description: "Represents a Concept Activation Vector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents a Concept Activation Vector.

## For Beginners

A CAV is a vector in the layer activation space that
points in the direction of "more concept." It's the key component of TCAV
that allows us to measure concept influence on predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConceptActivationVector(Vector<>,Double)` | Initializes a new Concept Activation Vector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassifierAccuracy` | Gets the accuracy of the linear classifier used to train this CAV. |
| `Weights` | Gets the CAV weights (direction in activation space). |

