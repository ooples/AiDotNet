---
title: "AudioClassifierBase<T>"
description: "Base class for audio classification models (genre, event detection, scene classification)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.Classification`

Base class for audio classification models (genre, event detection, scene classification).

## For Beginners

Audio classification is like teaching a computer to recognize
different types of sounds, similar to how you can tell the difference between
a dog barking and a car horn.

This base class provides:

- Class label management
- Softmax for probability conversion
- Common feature extraction

## How It Works

Audio classification assigns labels to audio clips. This base class provides
common functionality for various classification tasks including:

- Genre classification (rock, jazz, classical)
- Audio event detection (dog bark, car horn)
- Scene classification (office, park, street)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioClassifierBase(NeuralNetworkArchitecture<>)` | Initializes a new instance of the AudioClassifierBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassLabels` | Gets the list of class labels this model can classify. |
| `NumClasses` | Gets the number of classes this model can classify. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Tensor<>)` | Applies softmax to convert logits tensor to probabilities. |
| `ApplySoftmax(Vector<>)` | Applies softmax to convert logits to probabilities. |
| `ApplyThreshold(Dictionary<String,>,)` | Applies threshold for multi-label classification. |
| `ComputeClassWeights(Dictionary<String,Int32>)` | Computes class weights for imbalanced datasets. |
| `GetPrediction(Dictionary<String,>)` | Gets the predicted class (highest probability). |
| `GetTopK(Dictionary<String,>,Int32)` | Gets the top-K predictions sorted by probability. |

