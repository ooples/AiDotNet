---
title: "ZOClientMessage"
description: "Compact message from a client in the FedMeZO protocol."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Compact message from a client in the FedMeZO protocol.
Contains only the loss difference and seed — the server reconstructs the gradient.

## For Beginners

Instead of sending millions of parameter values, each client
sends just two numbers: how much the loss changed when we wiggled the model in a random
direction (lossDifference) and which random direction we used (seed). The server can
recreate the random direction from the seed and compute the full gradient estimate.
This makes communication ~1,000,000x cheaper.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZOClientMessage(Double,Int32)` | Creates a new ZO client message. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LossDifference` | The loss difference: L(w + epsilon*z) - L(w - epsilon*z). |
| `Seed` | The random seed that generates the perturbation vector z. |

