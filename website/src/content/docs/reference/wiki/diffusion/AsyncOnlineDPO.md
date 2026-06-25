---
title: "AsyncOnlineDPO<T>"
description: "Asynchronous Online DPO for diffusion model alignment with on-policy generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Alignment`

Asynchronous Online DPO for diffusion model alignment with on-policy generation.

## For Beginners

Regular DPO learns from a fixed dataset of preferred/dispreferred image
pairs collected beforehand. Async Online DPO is smarter — it continuously generates new images
with the current model, gets feedback on them, and learns from that fresh feedback. It's like
a student who keeps practicing and getting real-time feedback rather than studying old examples.

## How It Works

Async Online DPO extends standard DPO by generating new preference pairs during training
rather than relying on a static dataset. It asynchronously generates image pairs from the
current policy, obtains preference labels (from a reward model or human feedback), and
uses these fresh pairs for DPO updates. This on-policy approach leads to better alignment
than offline DPO trained on stale data.

Reference: Calandriello et al., "Human Alignment of Large Language Models through Online
Preference Optimization", 2024; adapted for diffusion models

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AsyncOnlineDPO(IDiffusionModel<>,IDiffusionModel<>,Double,Double,Int32)` | Initializes a new Async Online DPO trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta` | Gets the temperature parameter. |
| `OnlineMixingRatio` | Gets the ratio of online vs offline samples used in each batch. |
| `TotalUpdates` | Gets the total number of DPO updates performed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOnlineDPOLoss(,,,,Boolean)` | Computes the online DPO loss combining online and offline preference pairs. |
| `RecordUpdate` | Increments the update counter after a training step. |
| `ShouldRefreshReference(Int32)` | Determines if the reference model should be refreshed based on update count. |

