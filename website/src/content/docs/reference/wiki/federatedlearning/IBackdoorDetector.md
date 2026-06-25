---
title: "IBackdoorDetector<T>"
description: "Interface for detecting backdoor attacks in federated learning updates."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.BackdoorDefense`

Interface for detecting backdoor attacks in federated learning updates.

## For Beginners

Imagine a stop sign recognition model. A backdoor attack might make
the model correctly identify all normal stop signs, but misclassify any stop sign with a
small yellow sticker as a speed limit sign. Standard defenses (like Krum or Bulyan) may
not catch this because the poisoned update looks statistically normal overall — it only
misbehaves on the specific trigger pattern.

## How It Works

Backdoor attacks are a stealthy form of poisoning where a malicious client injects a
"trigger pattern" into the model. The model behaves normally on clean inputs but produces
attacker-chosen outputs when the trigger is present. Unlike untargeted poisoning (handled
by Byzantine-robust aggregators), backdoor attacks are targeted and can evade statistical
anomaly detection.

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectorName` | Gets the detector name for logging. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectSuspiciousUpdates(Dictionary<Int32,Vector<>>,Vector<>)` | Analyzes client updates and returns a suspicion score for each client. |
| `FilterMaliciousUpdates(Dictionary<Int32,Vector<>>,Vector<>,Double)` | Filters out suspected backdoor updates, returning only clean updates. |

