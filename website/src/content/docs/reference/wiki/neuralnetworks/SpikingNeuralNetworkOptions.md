---
title: "SpikingNeuralNetworkOptions"
description: "Configuration options for the SpikingNeuralNetwork."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NeuralNetworks.Options`

Configuration options for the SpikingNeuralNetwork.

## Properties

| Property | Summary |
|:-----|:--------|
| `ReadoutLearningRate` | Learning rate for the supervised surrogate-gradient delta-rule at the output layer and the unsupervised STDP updates at the hidden layers. |
| `StdpWindow` | STDP time window (number of time steps to consider for spike-timing correlations) applied by the unsupervised pair-based STDP learning rule (Gerstner & Kistler 2002). |

