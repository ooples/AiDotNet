---
title: "NeuralCleanseDetector<T>"
description: "Neural Cleanse — post-hoc backdoor detection by reverse-engineering potential triggers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.BackdoorDefense`

Neural Cleanse — post-hoc backdoor detection by reverse-engineering potential triggers.

## For Beginners

This detector works backwards — instead of looking at training updates,
it asks: "Is there a tiny pattern I can add to any input that makes the model always predict
a specific class?" If yes, someone probably planted a backdoor. For each possible target class,
it finds the smallest such pattern. If one class has an unusually small trigger, that's the
backdoor target class.

## How It Works

Neural Cleanse (Wang et al., 2019) detects backdoors by searching for the smallest
perturbation (trigger) that causes the model to misclassify inputs to a target class.
If such a small trigger exists for any class, the model is likely backdoored.
The anomaly index measures how much smaller the trigger for one class is compared to others.

Reference: Wang et al. (2019), "Neural Cleanse: Identifying and Mitigating Backdoor Attacks
in Neural Networks" (IEEE S&P 2019).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralCleanseDetector(Int32,Double)` | Creates a new Neural Cleanse detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectorName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectSuspiciousUpdates(Dictionary<Int32,Vector<>>,Vector<>)` |  |
| `FilterMaliciousUpdates(Dictionary<Int32,Vector<>>,Vector<>,Double)` |  |

