---
title: "NeuralNetworkOptions"
description: "Base configuration options for all neural network models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Base configuration options for all neural network models.

## For Beginners

This contains the basic settings that all neural networks share.
More specific neural network types (audio, document, financial, etc.) extend this with
domain-specific settings.

## How It Works

This class provides the foundational options shared by all neural network models,
inheriting the Seed property from ModelOptions. Neural network-specific options
like learning rate, epochs, and batch size can be added here as the library evolves.

## Properties

| Property | Summary |
|:-----|:--------|
| `EncoderLayerCount` | When providing custom layers via Architecture.Layers, specifies where the encoder ends and the decoder begins. |

