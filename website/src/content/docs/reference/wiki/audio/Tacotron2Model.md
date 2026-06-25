---
title: "Tacotron2Model<T>"
description: "Tacotron2 attention-based text-to-speech model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.TextToSpeech`

Tacotron2 attention-based text-to-speech model.

## For Beginners

Tacotron2 is a two-stage TTS system:

Stage 1 (Tacotron2): Text -> Mel Spectrogram
Stage 2 (Vocoder): Mel Spectrogram -> Audio Waveform

Key characteristics:

- Autoregressive: Generates one mel frame at a time
- Attention-based: Learns to align text with audio
- High quality but slower than parallel models like VITS

Two ways to use this class:

1. ONNX Mode: Load pretrained Tacotron2 models for inference
2. Native Mode: Train your own TTS model from scratch

ONNX Mode Example:

Training Mode Example:

## How It Works

Tacotron2 is a classic neural TTS model that generates mel spectrograms from text.
It uses an encoder-attention-decoder architecture with:

- Character/phoneme encoder with convolutional layers
- Location-sensitive attention for alignment
- Autoregressive LSTM decoder
- Post-net for mel spectrogram refinement

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Tacotron2Model(NeuralNetworkArchitecture<>,Int32,Int32,Double,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Tacotron2ModelOptions)` | Creates a Tacotron2 model for native training mode. |
| `Tacotron2Model(NeuralNetworkArchitecture<>,String,String,Int32,Int32,Double,Int32,Double,Int32,Int32,Int32,OnnxModelOptions,Tacotron2ModelOptions)` | Creates a Tacotron2 model for ONNX inference with pretrained models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableVoices` | Gets the list of available built-in voices. |
| `IsReady` | Gets whether the model is ready for synthesis. |
| `MaxDecoderSteps` | Gets the maximum decoder steps. |
| `SupportsEmotionControl` | Gets whether this model supports emotional expression control. |
| `SupportsStreaming` | Gets whether this model supports streaming audio generation. |
| `SupportsVoiceCloning` | Gets whether this model supports voice cloning from reference audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes the model and releases resources. |
| `ExtractSpeakerEmbedding(Tensor<>)` | Extracts speaker embedding from reference audio. |
| `ForwardForTraining(Tensor<>)` | Overrides ForwardForTraining to use teacher forcing when target is available. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes layers for ONNX inference mode. |
| `InitializeNativeLayers` | Initializes layers for native training mode. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `StartStreamingSession(String,Double)` | Starts a streaming synthesis session. |
| `Synthesize(String,String,Double,Double)` | Synthesizes speech from text. |
| `SynthesizeAsync(String,String,Double,Double,CancellationToken)` | Synthesizes speech from text asynchronously. |
| `SynthesizeWithEmotion(String,String,Double,String,Double)` | Synthesizes speech with emotional expression. |
| `SynthesizeWithVoiceCloning(String,Tensor<>,Double,Double)` | Synthesizes speech using a cloned voice from reference audio. |
| `UpdateParameters(Vector<>)` | Updates model parameters using the configured optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_acousticModel` | ONNX acoustic model (Tacotron2). |
| `_acousticModelPath` | Path to the acoustic model ONNX file. |
| `_attentionDim` | Attention dimension. |
| `_attentionFilters` | Attention location filters. |
| `_attentionLayers` | Attention layers. |
| `_decoderDim` | Decoder hidden dimension. |
| `_decoderLstmLayers` | Decoder LSTM layers. |
| `_disposed` | Whether the model has been disposed. |
| `_embedding` | Character/phoneme embedding layer. |
| `_embeddingDim` | Embedding dimension. |
| `_encoderConvLayers` | Encoder convolutional layers. |
| `_encoderDim` | Encoder hidden dimension. |
| `_encoderLstm` | Encoder LSTM layer. |
| `_fftSize` | FFT size for Griffin-Lim. |
| `_griffinLim` | Griffin-Lim vocoder fallback. |
| `_griffinLimIterations` | Griffin-Lim iterations. |
| `_hopLength` | Hop length for audio synthesis. |
| `_lossFunction` | Loss function for training. |
| `_maxDecoderSteps` | Maximum decoder steps. |
| `_numEncoderConvLayers` | Number of encoder convolutional layers. |
| `_numMelsPerFrame` | Number of mel frames to output per decoder step. |
| `_numPostnetConvLayers` | Number of post-net convolutional layers. |
| `_optimizer` | Optimizer for training. |
| `_postNetLayers` | Post-net layers for mel refinement. |
| `_postnetEmbeddingDim` | Post-net embedding dimension. |
| `_prenetDim` | Pre-net dimension. |
| `_preprocessor` | Text preprocessor for phoneme conversion. |
| `_speakingRate` | Speaking rate multiplier. |
| `_stopThreshold` | Decoder stop threshold. |
| `_stopTokenLayer` | Stop token prediction layer. |
| `_teacherForcingTarget` | Trains the model on input data. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX models (false). |
| `_vocabSize` | Character/phoneme vocabulary size. |
| `_vocoder` | ONNX vocoder model (HiFi-GAN or WaveGlow). |
| `_vocoderPath` | Path to the vocoder ONNX file. |

