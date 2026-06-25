---
title: "ISceneClassifier<T>"
description: "Interface for acoustic scene classification models that identify the environment/context of audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for acoustic scene classification models that identify the environment/context of audio.

## For Beginners

Scene classification is like asking "Where was this recording made?"

How it works:

1. Audio features capture the overall acoustic character
2. A classifier matches these features to known scene types
3. The most likely scene (and alternatives) are returned

Example scenes:

- Indoor: Office, restaurant, kitchen, library, shopping mall
- Outdoor: Park, street, beach, forest, construction site
- Transportation: Car, bus, train, metro, airport

How scenes differ from events:

- Event: "A dog barked" (specific sound)
- Scene: "This was recorded in a park" (overall environment)

Use cases:

- Context-aware devices (adjust phone behavior based on location)
- Audio organization (group recordings by location)
- Surveillance (detect unusual environments)
- AR/VR (match virtual audio to real environment)
- Assistive technology (describe environment to blind users)

## How It Works

Acoustic scene classification (ASC) identifies the environment or context where audio was recorded.
Unlike event detection which finds specific sounds, scene classification characterizes the overall
acoustic atmosphere.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MinimumDurationSeconds` | Gets the minimum audio duration required for reliable classification. |
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportedScenes` | Gets the list of scenes this model can classify. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Tensor<>)` | Classifies the acoustic scene of audio. |
| `ClassifyAsync(Tensor<>,CancellationToken)` | Classifies acoustic scene asynchronously. |
| `ExtractAcousticFeatures(Tensor<>)` | Extracts acoustic features used for scene classification. |
| `GetSceneProbabilities(Tensor<>)` | Gets scene probabilities for all supported scenes. |
| `GetTopScenes(Tensor<>,Int32)` | Gets top-K scene predictions. |
| `TrackSceneChanges(Tensor<>,Double)` | Tracks scene changes over time in longer audio. |

