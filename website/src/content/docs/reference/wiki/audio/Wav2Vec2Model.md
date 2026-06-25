---
title: "Wav2Vec2Model<T>"
description: "Wav2Vec2 self-supervised speech recognition model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SpeechRecognition`

Wav2Vec2 self-supervised speech recognition model.

## For Beginners

Wav2Vec2 works differently from traditional speech recognition:

1. It processes raw audio directly (no mel spectrograms needed)
2. It learns speech patterns from unlabeled audio data
3. It can be fine-tuned with small amounts of labeled data

Architecture:

- Convolutional feature encoder: Processes raw audio into features
- Transformer encoder: Captures long-range dependencies in speech
- CTC head: Aligns speech to text (Connectionist Temporal Classification)

Two ways to use this class:

1. ONNX Mode: Load pretrained Wav2Vec2 models for fast inference
2. Native Mode: Train your own speech recognition model from scratch

ONNX Mode Example:

Training Mode Example:

## How It Works

Wav2Vec2 is a self-supervised learning model for speech recognition developed by Meta AI.
It learns representations from raw audio through contrastive learning, then can be
fine-tuned for speech recognition tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Wav2Vec2Model(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,String[],IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Wav2Vec2ModelOptions)` | Creates a Wav2Vec2 network for training from scratch using native layers. |
| `Wav2Vec2Model(NeuralNetworkArchitecture<>,String,String,Int32,Int32,String[],OnnxModelOptions,Wav2Vec2ModelOptions)` | Creates a Wav2Vec2 network using a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` | Gets whether the model is ready for inference. |
| `Language` | Gets the target language for transcription. |
| `MaxAudioLengthSeconds` | Gets the maximum audio length in seconds. |
| `SupportedLanguages` | Gets the list of languages supported by this model. |
| `SupportsStreaming` | Gets whether this model supports real-time streaming transcription. |
| `SupportsWordTimestamps` | Gets whether this model can identify timestamps for each word. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `DetectLanguage(Tensor<>)` | Detects the language spoken in the audio. |
| `DetectLanguageProbabilities(Tensor<>)` | Gets language detection probabilities for the audio. |
| `Dispose(Boolean)` | Disposes the model and releases resources. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes layers for ONNX inference mode. |
| `InitializeNativeLayers` | Initializes native mode layers for training from scratch. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `StartStreamingSession(String)` | Starts a streaming transcription session. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single batch. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio to text. |
| `TranscribeAsync(Tensor<>,String,Boolean,CancellationToken)` | Transcribes audio to text asynchronously. |
| `UpdateParameters(Vector<>)` | Updates model parameters by applying gradient descent. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_ctcProjection` | CTC projection layer. |
| `_disposed` | Disposed flag. |
| `_featureEncoderLayers` | Convolutional feature encoder layers. |
| `_ffDim` | Feed-forward dimension (non-readonly for deserialization support). |
| `_hiddenDim` | Hidden dimension for the transformer (non-readonly for deserialization support). |
| `_language` | Target language for transcription (non-readonly for deserialization support). |
| `_lossFunction` | Loss function for training. |
| `_maxAudioLengthSeconds` | Maximum audio length in seconds (non-readonly for deserialization support). |
| `_modelPath` | Path to the ONNX model file. |
| `_numHeads` | Number of attention heads (non-readonly for deserialization support). |
| `_numTransformerLayers` | Number of transformer layers (non-readonly for deserialization support). |
| `_optimizer` | Optimizer for training (unused in ONNX mode). |
| `_transformerLayers` | Transformer encoder layers. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX models (false). |
| `_vocabSize` | Vocabulary size for CTC output (non-readonly for deserialization support). |
| `_vocabulary` | Vocabulary mapping for CTC decoding (non-readonly for deserialization support). |

