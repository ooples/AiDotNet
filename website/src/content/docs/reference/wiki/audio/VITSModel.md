---
title: "VITSModel<T>"
description: "VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.TextToSpeech`

VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) model.

## For Beginners

VITS is a modern TTS model with several advantages:

1. End-to-end: Converts text directly to audio (no separate vocoder needed)
2. Fast: Parallel generation is much faster than autoregressive models
3. High quality: Produces natural-sounding speech
4. Voice cloning: Can learn to speak in new voices from short audio samples

Two ways to use this class:

1. ONNX Mode: Load pretrained VITS models for fast inference
2. Native Mode: Train your own TTS model from scratch

ONNX Mode Example:

Voice Cloning Example:

## How It Works

VITS is a state-of-the-art end-to-end TTS model that generates high-quality speech
directly from text without requiring a separate vocoder. It combines:

- Variational autoencoder (VAE) for learning latent representations
- Normalizing flows for improved audio quality
- Adversarial training for realistic speech synthesis
- Multi-speaker support with speaker embeddings

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VITSModel(NeuralNetworkArchitecture<>,Int32,Int32,Double,Double,Double,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32[],Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,VITSModelOptions)` | Creates a VITS model for native training mode. |
| `VITSModel(NeuralNetworkArchitecture<>,String,String,Int32,Int32,Double,Double,Double,Int32,Int32,OnnxModelOptions,VITSModelOptions)` | Creates a VITS model for ONNX inference with pretrained models. |

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
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `UpdateParameters(Vector<>)` | Updates model parameters using the configured optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultPhonemeVocabSize` | Default phoneme vocabulary size for character-level models. |
| `DefaultUpsampleRates` | Default HiFi-GAN upsample rates producing 256x upsampling (8*8*2*2). |
| `_decoderLayers` | Decoder layers (HiFi-GAN style generator). |
| `_disposed` | Whether the model has been disposed. |
| `_durationPredictorLayers` | Duration predictor layers. |
| `_fftSize` | FFT size for audio generation. |
| `_flowLayers` | Flow layers for normalizing flows. |
| `_hiddenDim` | Hidden dimension for the model. |
| `_hopLength` | Hop length for audio generation. |
| `_lengthScale` | Length scale for duration control. |
| `_lossFunction` | Loss function for training. |
| `_maxPhonemeLength` | Maximum phoneme sequence length. |
| `_melSpectrogram` | Mel spectrogram extractor for speaker encoding. |
| `_modelPath` | Path to the ONNX model file. |
| `_noiseScale` | Noise scale for sampling. |
| `_numEncoderLayers` | Number of text encoder layers. |
| `_numFlowLayers` | Number of flow layers. |
| `_numHeads` | Number of attention heads. |
| `_numSpeakers` | Number of speakers for multi-speaker model. |
| `_optimizer` | Optimizer for training. |
| `_phonemeVocabSize` | Phoneme vocabulary size. |
| `_preprocessor` | Text preprocessor for phoneme conversion. |
| `_speakerEmbedding` | Speaker embedding layer. |
| `_speakerEmbeddingDim` | Speaker embedding dimension. |
| `_speakerEncoder` | ONNX speaker encoder model. |
| `_speakerEncoderPath` | Path to the speaker encoder ONNX model (for voice cloning). |
| `_speakingRate` | Speaking rate multiplier. |
| `_textEncoderLayers` | Text encoder layers. |
| `_upsampleRates` | HiFi-GAN upsampling rates for the decoder. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX models (false). |

