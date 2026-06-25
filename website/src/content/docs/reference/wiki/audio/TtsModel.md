---
title: "TtsModel<T>"
description: "Text-to-speech model for synthesizing speech from text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.TextToSpeech`

Text-to-speech model for synthesizing speech from text.

## For Beginners

Text-to-Speech works like this:

1. Your text is converted to phonemes (speech sounds)
2. The acoustic model predicts what the speech should "look like" (mel spectrogram)
3. The vocoder makes it actually sound like speech

This class supports two modes:

- ONNX Mode: Load pretrained FastSpeech2/HiFi-GAN models for instant synthesis
- Native Mode: Train your own TTS model from scratch

Usage (ONNX Mode):

Usage (Native Training Mode):

## How It Works

This TTS model uses a two-stage pipeline:

1. Acoustic Model (FastSpeech2): Converts text/phonemes to mel spectrogram
2. Vocoder (HiFi-GAN or Griffin-Lim): Converts mel spectrogram to audio waveform

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TtsModel(NeuralNetworkArchitecture<>,Int32,Int32,Double,Double,Double,Nullable<Int32>,String,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,TtsOptions)` | Creates a TtsModel for native training mode. |
| `TtsModel(NeuralNetworkArchitecture<>,String,String,Int32,Int32,Double,Double,Double,Nullable<Int32>,String,Boolean,Int32,Int32,Int32,OnnxModelOptions,TtsOptions)` | Creates a TtsModel for ONNX inference with pretrained models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableVoices` | Gets the list of available built-in voices. |
| `IsReady` | Gets whether the model is ready for synthesis. |
| `SupportsEmotionControl` | Gets whether this model supports emotional expression control. |
| `SupportsStreaming` | Gets whether this model supports streaming audio generation. |
| `SupportsVoiceCloning` | Gets whether this model supports voice cloning from reference audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes the model and releases resources. |
| `ExtractSpeakerEmbedding(Tensor<>)` | Extracts speaker embedding from reference audio for voice cloning. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers for the TTS model. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `StartStreamingSession(String,Double)` | Starts a streaming synthesis session. |
| `Synthesize(String,String,Double,Double)` | Synthesizes speech from text. |
| `SynthesizeAsync(String,String,Double,Double,CancellationToken)` | Synthesizes speech from text asynchronously. |
| `SynthesizeWithEmotion(String,String,Double,String,Double)` | Synthesizes speech with emotional expression. |
| `SynthesizeWithVoiceCloning(String,Tensor<>,Double,Double)` | Synthesizes speech using a cloned voice from reference audio. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `UpdateParameters(Vector<>)` | Updates model parameters using gradient descent. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_acousticModel` | ONNX acoustic model (FastSpeech2 or similar). |
| `_acousticModelPath` | Path to the acoustic model ONNX file. |
| `_disposed` | Whether the model has been disposed. |
| `_energy` | Energy/volume level. |
| `_fftSize` | FFT size for Griffin-Lim. |
| `_griffinLim` | Griffin-Lim vocoder fallback. |
| `_griffinLimIterations` | Number of Griffin-Lim iterations. |
| `_hiddenDim` | Hidden dimension for the acoustic model. |
| `_hopLength` | Hop length for Griffin-Lim. |
| `_language` | Language code for multi-lingual models. |
| `_lossFunction` | Loss function for training. |
| `_maxPhonemeLength` | Maximum phoneme sequence length. |
| `_numDecoderLayers` | Number of decoder layers. |
| `_numEncoderLayers` | Number of encoder layers. |
| `_numHeads` | Number of attention heads. |
| `_optimizer` | Optimizer for training. |
| `_pitchShift` | Pitch shift in semitones. |
| `_preprocessor` | Text preprocessor for phoneme conversion. |
| `_speakerId` | Speaker ID for multi-speaker models. |
| `_speakingRate` | Speaking rate multiplier. |
| `_useGriffinLimFallback` | Whether to use Griffin-Lim as fallback vocoder. |
| `_useNativeMode` | Whether the model is operating in native training mode. |
| `_vocoder` | ONNX vocoder model (HiFi-GAN or similar). |
| `_vocoderModelPath` | Path to the vocoder model ONNX file. |

