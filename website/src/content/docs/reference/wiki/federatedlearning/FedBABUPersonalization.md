---
title: "FedBABUPersonalization<T>"
description: "Implements FedBABU (Body And Bottom Update) personalization strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements FedBABU (Body And Bottom Update) personalization strategy.

## For Beginners

FedBABU takes a deliberately simple approach to personalization:
during federated training, only the model body (feature extractor) is trained and aggregated,
while the classification head is frozen at random initialization. After FL converges, each
client locally fine-tunes just the head on their own data. This works surprisingly well
because a good feature extractor transfers across clients, and a few local epochs on the
head are enough for personalization.

## How It Works

Algorithm:

Reference: Oh, J., et al. (2022). "FedBABU: Toward Enhanced Representation for
Federated Image Classification." ICLR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedBABUPersonalization(Double,Int32)` | Creates a new FedBABU personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadFraction` | Gets the head fraction. |
| `LocalFineTuneEpochs` | Gets the local fine-tuning epochs. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBody(Dictionary<String,[]>)` | Extracts the body parameters (shared/aggregated) from the full model. |
| `ExtractHead(Dictionary<String,[]>)` | Extracts the head parameters (frozen during FL, fine-tuned locally after convergence). |
| `InitializeRandomHead(Dictionary<String,[]>,Int32)` | Initializes the classification head with random values (Kaiming uniform initialization). |
| `MaskHeadGradients(Dictionary<String,[]>)` | Applies a gradient mask that zeros out head gradients during FL body training. |
| `MergeBodyAndHead(Dictionary<String,[]>,Dictionary<String,[]>)` | Merges aggregated body with local head parameters. |

