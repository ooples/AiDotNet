---
title: "BackdoorDefenseOptions"
description: "Configuration options for backdoor attack detection and mitigation in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for backdoor attack detection and mitigation in federated learning.

## For Beginners

Backdoor attacks are stealthy — the model works correctly on clean data
but misbehaves when a specific trigger pattern is present. Unlike Byzantine attacks (which degrade
overall performance), backdoors are targeted and hard to detect statistically. These detectors
analyze client updates for tell-tale signs of backdoor injection.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignmentThreshold` | Gets or sets the cosine similarity threshold below which a subspace is suspicious. |
| `AnomalyThreshold` | Gets or sets the MAD-based anomaly threshold for Neural Cleanse. |
| `NumClasses` | Gets or sets the number of output classes for Neural Cleanse trigger analysis. |
| `NumSubspaces` | Gets or sets the number of parameter subspaces for direction alignment analysis. |
| `Strategy` | Gets or sets the detection strategy. |
| `SuspicionThreshold` | Gets or sets the suspicion threshold above which a client is filtered out. |

