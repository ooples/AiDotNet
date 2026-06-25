---
title: "ICertifiedDefense<T, TInput, TOutput>"
description: "Defines the contract for certified defense mechanisms that provide provable robustness guarantees."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for certified defense mechanisms that provide provable robustness guarantees.

## How It Works

Certified defenses provide mathematical guarantees that a model's predictions won't change
within a specified perturbation radius, unlike heuristic defenses.

**For Beginners:** Think of certified defenses as "guaranteed protection" for your model.
While regular defenses make models harder to fool, certified defenses can mathematically prove
that no attack within certain limits can trick the model.

Common certified defense methods include:

- Randomized Smoothing: Uses random noise to create certified predictions
- Interval Bound Propagation: Tracks ranges of possible values through the network
- CROWN: Computes certified bounds for neural network outputs

Why certified defenses matter:

- They provide provable security guarantees
- They're essential for safety-critical applications
- They help meet regulatory requirements
- They give confidence bounds for predictions

## Methods

| Method | Summary |
|:-----|:--------|
| `CertifyBatch([],IFullModel<,,>)` | Computes certified predictions for a batch of inputs. |
| `CertifyPrediction(,IFullModel<,,>)` | Computes a certified prediction with robustness guarantees. |
| `ComputeCertifiedRadius(,IFullModel<,,>)` | Computes the maximum perturbation radius that can be certified for an input. |
| `EvaluateCertifiedAccuracy([],[],IFullModel<,,>,)` | Evaluates certified accuracy on a dataset. |
| `GetOptions` | Gets the configuration options for the certified defense. |
| `Reset` | Resets the certified defense state. |

