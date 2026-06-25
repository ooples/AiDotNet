---
title: "SpeakerVerifier<T>"
description: "Verifies speaker identity by comparing embeddings against enrolled speakers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

Verifies speaker identity by comparing embeddings against enrolled speakers.

## For Beginners

Speaker verification is like voice-based password checking:

1. First, you "enroll" a speaker by recording their voice samples
2. Later, when someone claims to be that person, you record them and compare
3. If the voices match closely enough, the identity is verified

Usage (ONNX Mode):

Usage (Native Training Mode):

## How It Works

Speaker verification answers the question "Is this the person they claim to be?"
by comparing a test utterance against enrolled speaker embeddings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeakerVerifier` | Creates a SpeakerVerifier with default settings for native training mode. |
| `SpeakerVerifier(NeuralNetworkArchitecture<>,Int32,Int32,Double,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a SpeakerVerifier for native training mode. |
| `SpeakerVerifier(NeuralNetworkArchitecture<>,String,Int32,Int32,Double,OnnxModelOptions)` | Creates a SpeakerVerifier for ONNX inference with a pretrained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultThreshold` | Gets the default verification threshold. |
| `EmbeddingExtractor` | Gets the underlying speaker embedding extractor. |
| `EnrolledCount` | Gets the number of enrolled speakers. |
| `IsOnnxMode` | Gets whether the model is in ONNX inference mode. |
| `VerificationThreshold` | Gets the verification threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScore(Tensor<>,Tensor<>)` | Computes the verification score between audio and a reference. |
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes the model and releases resources. |
| `Enroll(IReadOnlyList<Tensor<>>)` | Enrolls a speaker by creating a reference embedding from audio samples. |
| `Enroll(String,SpeakerEmbedding<>[])` | Enrolls a speaker with one or more embeddings (legacy API). |
| `Enroll(Tensor<>)` | Enrolls a speaker by creating a reference embedding from a single audio sample. |
| `GetEnrolledSpeakers` | Gets all enrolled speaker IDs. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetThresholdForFAR(Double)` | Gets the recommended threshold for a target false accept rate. |
| `Identify(SpeakerEmbedding<>)` | Identifies the most likely speaker from enrolled set (legacy API). |
| `InitializeLayers` | Initializes the layers for the speaker verifier. |
| `IsEnrolled(String)` | Checks if a speaker is enrolled. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `Unenroll(String)` | Removes a speaker's enrollment. |
| `UpdateParameters(Vector<>)` | Updates model parameters. |
| `UpdateProfile(SpeakerProfile<>,Tensor<>)` | Updates an existing speaker profile with additional audio. |
| `Verify(String,SpeakerEmbedding<>)` | Verifies if a test embedding matches an enrolled speaker (legacy API). |
| `Verify(Tensor<>,Tensor<>)` | Verifies if audio matches a reference speaker embedding. |
| `Verify(Tensor<>,Tensor<>,)` | Verifies if audio matches a reference speaker embedding with custom threshold. |
| `VerifyAsync(Tensor<>,Tensor<>,CancellationToken)` | Verifies if audio matches a reference speaker embedding asynchronously. |
| `VerifyWithReferenceAudio(Tensor<>,Tensor<>)` | Verifies if audio matches reference audio of a claimed speaker. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_disposed` | Whether the model has been disposed. |
| `_embeddingExtractor` | Speaker embedding extractor. |
| `_embeddingModelPath` | Path to the speaker embedding model ONNX file. |
| `_enrolledSpeakers` | Enrolled speakers dictionary. |
| `_hiddenDim` | Hidden dimension for encoder layers. |
| `_lossFunction` | Loss function for training. |
| `_numEncoderLayers` | Number of encoder layers. |
| `_numHeads` | Number of attention heads. |
| `_optimizer` | Optimizer for training. |
| `_useNativeMode` | Whether the model is operating in native training mode. |

