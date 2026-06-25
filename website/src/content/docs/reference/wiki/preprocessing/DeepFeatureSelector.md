---
title: "DeepFeatureSelector<T>"
description: "Deep Feature Selection using a multi-layer neural network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Neural`

Deep Feature Selection using a multi-layer neural network.

## For Beginners

This method trains a multi-layer neural network
but penalizes the input layer weights. Features with larger input weights
after training are considered more important. The network can capture
complex non-linear patterns that simpler methods might miss.

## How It Works

Uses a deep neural network with L1 regularization on the first layer
to perform feature selection while learning non-linear relationships.

