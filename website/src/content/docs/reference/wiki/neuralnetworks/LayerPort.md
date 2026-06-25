---
title: "LayerPort"
description: "Declares a named input or output port on a layer."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Declares a named input or output port on a layer.
Ports enable multi-input layers (e.g., DiffusionResBlock needs "input" + "time_embed")
and provide compile-time documentation of a layer's data contract.

## For Beginners

A port is like a labeled plug on the layer.
Just as a TV has separate ports for HDMI, USB, and power, a neural network layer
can have separate ports for different types of input data.

