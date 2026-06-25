---
title: "WhisperModel<T>"
description: "Whisper automatic speech recognition model for transcribing audio to text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Whisper`

Whisper automatic speech recognition model for transcribing audio to text.

## For Beginners

Whisper converts spoken audio into text. It works by:

1. Converting audio to a mel spectrogram (visual representation of sound)
2. Processing through an encoder neural network
3. Generating text tokens through a decoder neural network

Two ways to use this class:

1. ONNX Mode: Load pretrained models for fast inference
2. Native Mode: Train your own speech recognition model from scratch

ONNX Mode Example:

Training Mode Example:

## How It Works

Whisper is a state-of-the-art speech recognition model by OpenAI that can:

- Transcribe speech in 99+ languages
- Translate non-English speech to English
- Detect the spoken language automatically
- Handle noisy audio and accents well

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WhisperModel(NeuralNetworkArchitecture<>,String,String,WhisperModelSize,String,Boolean,Int32,Int32,Int32,Int32,Int32,Double,OnnxModelOptions,WhisperOptions)` | Creates a Whisper network using pretrained ONNX models. |
| `WhisperModel(NeuralNetworkArchitecture<>,WhisperModelSize,String,Boolean,Int32,Int32,Int32,Int32,Int32,Double,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,WhisperOptions)` | Creates a Whisper network for training from scratch using native layers. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` | Gets whether the model is ready for inference. |
| `Language` | Gets the target language for transcription. |
| `MaxAudioLengthSeconds` | Gets the maximum audio length in seconds. |
| `ModelSize` | Gets the model size variant. |
| `SupportedLanguages` | Gets the list of languages supported by this model. |
| `SupportsStreaming` | Gets whether this model supports real-time streaming transcription. |
| `SupportsWordTimestamps` | Gets whether this model can identify timestamps for each word. |
| `Translate` | Gets whether translation to English is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `DetectLanguage(Tensor<>)` | Detects the language spoken in the audio. |
| `DetectLanguageProbabilities(Tensor<>)` | Gets language detection probabilities for the audio. |
| `Dispose(Boolean)` | Disposes the model and releases resources. |
| `ExtractSegments(List<Int64>,String)` | Segments a Whisper token stream into time-aligned text spans by parsing the timestamp tokens the model emits between transcribed phrases (Whisper §2.3 — timestamp tokens are interleaved with text tokens at the `TimestampBeginId + n` offsets… |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes layers following the golden standard pattern. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `StartStreamingSession(String)` | Starts a streaming transcription session. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single batch of audio and expected transcription tokens. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio to text. |
| `TranscribeAsync(Tensor<>,String,Boolean,CancellationToken)` | Transcribes audio to text asynchronously. |
| `UpdateParameters(Vector<>)` | Updates model parameters by applying gradient descent. |
| `ValidateLayerConfiguration(List<ILayer<>>)` | Validates that custom layers meet Whisper requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_beamSize` | Beam size for beam search decoding. |
| `_decoderPath` | Path to the decoder ONNX model file. |
| `_disposed` | Disposed flag. |
| `_encoderPath` | Path to the encoder ONNX model file. |
| `_ffDim` | Feed-forward dimension. |
| `_language` | Target language for transcription (null for auto-detect). |
| `_lossFunction` | Loss function for training. |
| `_maxAudioLengthSeconds` | Maximum audio length in seconds. |
| `_maxTokens` | Maximum number of tokens to generate. |
| `_melSpectrogram` | The mel spectrogram preprocessor. |
| `_modelDim` | Model dimension (hidden size). |
| `_modelSize` | Model size variant. |
| `_numDecoderLayers` | Number of decoder layers. |
| `_numEncoderLayers` | Number of encoder layers. |
| `_numHeads` | Number of attention heads. |
| `_numMels` | Number of mel filterbank channels. |
| `_optimizer` | Optimizer for training (unused in ONNX mode). |
| `_temperature` | Temperature for sampling (0 = greedy). |
| `_tokenizer` | The tokenizer for converting between text and tokens. |
| `_translate` | Whether to translate to English. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX models (false). |

