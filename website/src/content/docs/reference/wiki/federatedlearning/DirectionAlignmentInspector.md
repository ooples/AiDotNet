---
title: "DirectionAlignmentInspector<T>"
description: "Direction Alignment Inspector — detects backdoor attacks via gradient direction analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.BackdoorDefense`

Direction Alignment Inspector — detects backdoor attacks via gradient direction analysis.

## For Beginners

In normal training, all clients push the model in roughly the same
direction. A backdoor attacker pushes in a different direction for certain parts of the model
(the parts that encode the trigger). By checking whether each client's "push direction"
aligns with the majority, we can identify suspicious clients.

## How It Works

This detector identifies backdoor attacks by analyzing the alignment between each client's
update direction and the "honest" consensus direction. Backdoor updates tend to have
anomalous gradient directions that point away from the clean learning direction, particularly
in specific parameter subspaces corresponding to the trigger-target mapping.

Reference: Xu et al. (2025), "Detecting Backdoor Attacks in Federated Learning via Direction
Alignment Inspection" (CVPR 2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DirectionAlignmentInspector(Int32,Double)` | Creates a new Direction Alignment Inspector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectorName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectSuspiciousUpdates(Dictionary<Int32,Vector<>>,Vector<>)` |  |
| `FilterMaliciousUpdates(Dictionary<Int32,Vector<>>,Vector<>,Double)` |  |

