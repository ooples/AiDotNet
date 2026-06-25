---
title: "AudioGenModel<T>"
description: "AudioGen model for generating audio from text descriptions using neural audio codecs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.AudioGen`

AudioGen model for generating audio from text descriptions using neural audio codecs.

## For Beginners

AudioGen is fundamentally different from Text-to-Speech (TTS):

TTS vs AudioGen:

- TTS: Converts specific words to speech ("Hello world" -> spoken words "Hello world")
- AudioGen: Creates sounds matching a description ("dog barking" -> actual bark sound)

How it works:

1. Your text prompt ("a cat meowing softly") is encoded into a numerical representation
2. A language model generates a sequence of "audio tokens" (like words, but for sound)
3. The EnCodec decoder converts these tokens back into actual audio waveforms

Why discrete audio codes?

- Raw audio has too many samples (32,000 per second!)
- EnCodec compresses audio to ~50 tokens per second
- This makes the language model's job much easier

Common use cases:

- Sound effect generation for games/films
- Creating ambient soundscapes
- Generating audio for multimedia content
- Rapid prototyping of audio concepts

Limitations:

- Cannot generate intelligible speech (use TTS for that)
- Quality depends on training data
- May struggle with very specific or unusual sounds

## How It Works

AudioGen uses a language model approach to generate audio from text prompts.
The architecture consists of three main components:

- Text Encoder: Converts text prompts to embeddings (typically T5-based)
- Audio Language Model: Generates discrete audio codes autoregressively
- Audio Decoder (EnCodec): Converts audio codes back to waveforms

Reference: "AudioGen: Textually Guided Audio Generation" by Kreuk et al., 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioGenModel(NeuralNetworkArchitecture<>,AudioGenModelSize,Int32,Double,Double,Double,Int32,Double,Double,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Nullable<Int32>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,AudioGenOptions)` | Creates an AudioGen network using native library layers for training from scratch. |
| `AudioGenModel(NeuralNetworkArchitecture<>,String,String,String,ITokenizer,AudioGenModelSize,Int32,Double,Double,Double,Int32,Double,Double,Int32,Nullable<Int32>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,AudioGenOptions)` | Creates an AudioGen network using pretrained ONNX models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` | Gets whether the model is ready for inference. |
| `MaxDurationSeconds` | Gets the maximum duration of audio that can be generated in seconds. |
| `ModelSize` | Gets the model size variant. |
| `SupportsAudioContinuation` | Gets whether this model supports audio continuation. |
| `SupportsAudioInpainting` | Gets whether this model supports audio inpainting. |
| `SupportsTextToAudio` | Gets whether this model supports text-to-audio generation. |
| `SupportsTextToMusic` | Gets whether this model supports text-to-music generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` | Continues existing audio to extend it naturally. |
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DecodeAudioNative(Tensor<>,Double)` | Decodes audio codes to waveform in native mode using neural network layers. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes the model and releases resources. |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates audio from a text description. |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` | Generates audio from a text description asynchronously. |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates music from a text description. |
| `GetDefaultOptions` | Gets default generation options. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers following the golden standard pattern. |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` | Fills in missing or masked sections of audio. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `UpdateParameters(Vector<>)` | Updates model parameters by applying gradient descent. |
| `ValidateLayerConfiguration(List<ILayer<>>,Int32,Int32)` | Validates that custom layers meet AudioGen requirements and determines layer boundaries. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_audioDecoder` | The ONNX model for audio decoding (EnCodec). |
| `_audioDecoderPath` | Path to the audio decoder ONNX model file. |
| `_channels` | Number of audio channels (1=mono, 2=stereo). |
| `_codebookSize` | Size of each codebook vocabulary. |
| `_disposed` | Disposed flag. |
| `_durationSeconds` | Default duration of generated audio in seconds. |
| `_guidanceScale` | Classifier-free guidance scale. |
| `_languageModel` | The ONNX model for audio language modeling. |
| `_languageModelPath` | Path to the language model ONNX model file. |
| `_lmHiddenDim` | Language model hidden dimension. |
| `_lossFunction` | Loss function for training. |
| `_maxDurationSeconds` | Maximum duration of generated audio in seconds. |
| `_maxTextLength` | Maximum text sequence length. |
| `_modelSize` | Model size variant. |
| `_numCodebooks` | Number of EnCodec codebooks. |
| `_numHeads` | Number of attention heads. |
| `_numLmLayers` | Number of transformer layers in the language model. |
| `_optimizer` | Optimizer for training. |
| `_random` | Random number generator for sampling. |
| `_randomLock` | Lock object for thread-safe random access. |
| `_sampleRate` | Output sample rate in Hz. |
| `_temperature` | Sampling temperature (higher = more random). |
| `_textEncoder` | The ONNX model for text encoding. |
| `_textEncoderPath` | Path to the text encoder ONNX model file. |
| `_textHiddenDim` | Text encoder hidden dimension. |
| `_tokenizer` | The tokenizer for processing text input. |
| `_topK` | Top-k sampling parameter. |
| `_topP` | Top-p (nucleus) sampling parameter. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX models (false). |

