---
title: "ISpeakerVerifier<T>"
description: "Interface for speaker verification models that determine if audio matches a claimed identity."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for speaker verification models that determine if audio matches a claimed identity.

## For Beginners

Speaker verification is like a voice-based password check.

How it works:

1. User enrolls by providing voice samples
2. System creates a voiceprint (speaker embedding) for that user
3. Later, user provides a new voice sample
4. System compares new sample to stored voiceprint
5. Decision: Accept (same person) or Reject (different person)

Common use cases:

- Voice banking authentication
- Phone-based customer verification
- Smart speaker personalization
- Access control systems

Key metrics:

- False Accept Rate (FAR): How often imposters are wrongly accepted
- False Reject Rate (FRR): How often legitimate users are wrongly rejected
- Equal Error Rate (EER): When FAR = FRR (lower is better)

## How It Works

Speaker verification (also called speaker authentication) determines whether a speech
sample matches a claimed identity. It answers the question "Is this person who they
claim to be?" This is a 1-to-1 comparison task.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultThreshold` | Gets the default decision threshold for verification. |
| `EmbeddingExtractor` | Gets the underlying speaker embedding extractor. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `SampleRate` | Gets the expected sample rate for input audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScore(Tensor<>,Tensor<>)` | Computes the verification score between audio and a reference. |
| `Enroll(IReadOnlyList<Tensor<>>)` | Enrolls a speaker by creating a reference embedding from audio samples. |
| `Enroll(Tensor<>)` | Enrolls a speaker by creating a reference embedding from a single audio sample. |
| `GetThresholdForFAR(Double)` | Gets the recommended threshold for a target false accept rate. |
| `UpdateProfile(SpeakerProfile<>,Tensor<>)` | Updates an existing speaker profile with additional audio. |
| `Verify(Tensor<>,Tensor<>)` | Verifies if audio matches a reference speaker embedding. |
| `Verify(Tensor<>,Tensor<>,)` | Verifies if audio matches a reference speaker embedding with custom threshold. |
| `VerifyAsync(Tensor<>,Tensor<>,CancellationToken)` | Verifies if audio matches a reference speaker embedding asynchronously. |
| `VerifyWithReferenceAudio(Tensor<>,Tensor<>)` | Verifies if audio matches reference audio of a claimed speaker. |

