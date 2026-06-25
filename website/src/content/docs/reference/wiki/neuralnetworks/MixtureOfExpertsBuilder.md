---
title: "MixtureOfExpertsBuilder<T>"
description: "A builder class that helps create and configure Mixture-of-Experts layers with sensible defaults."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A builder class that helps create and configure Mixture-of-Experts layers with sensible defaults.

## For Beginners

Think of this as a guided recipe for creating an MoE layer.

Instead of manually specifying every detail of your MoE layer (which experts to use,
how to route between them, whether to use load balancing, etc.), this builder provides
good default choices based on research and best practices.

It's like having a cooking recipe that says "preheat to 350°F" instead of making you
figure out the right temperature yourself. You can still customize if needed, but the
defaults work well for most cases.

## How It Works

This builder simplifies the creation of Mixture-of-Experts layers by providing convenient methods
with research-backed default values. It follows best practices from MoE literature to ensure
good initial configuration for most use cases.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixtureOfExpertsBuilder` | Initializes a new instance of the `MixtureOfExpertsBuilder` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Build` | Builds the Mixture-of-Experts layer with the configured settings. |
| `CreateExpert` | Creates a single expert network based on the configured settings. |
| `WithDimensions(Int32,Int32)` | Sets the input and output dimensions for the MoE layer. |
| `WithExpertActivation(IActivationFunction<>)` | Sets the activation function for experts. |
| `WithExpertHiddenDim(Int32)` | Sets the hidden dimension for the expert networks (for 2-layer experts). |
| `WithExperts(Int32)` | Sets the number of expert networks in the MoE layer. |
| `WithHiddenExpansion(Int32)` | Sets the hidden dimension expansion factor for expert networks. |
| `WithIntermediateLayer(Boolean)` | Configures whether experts should use an intermediate (hidden) layer. |
| `WithLoadBalancing(Boolean,Double)` | Configures load balancing to encourage even expert utilization. |
| `WithOutputActivation(IActivationFunction<>)` | Sets the activation function for the MoE layer output. |
| `WithTopK(Int32)` | Configures Top-K sparse routing. |

