---
title: "SpeechEmotionRecognizer<T>"
description: "Neural network-based speech emotion recognition model that classifies emotional states from audio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Emotion`

Neural network-based speech emotion recognition model that classifies emotional states from audio.

## For Beginners

This is like teaching a computer to "hear" emotions in someone's voice!

How it works:

1. Audio is converted to a mel spectrogram (a visual representation of sound frequencies over time)
2. A neural network analyzes patterns in the spectrogram
3. The network outputs probabilities for each emotion (happy, sad, angry, etc.)

Key features detected:

- Pitch patterns (high pitch often = excitement, low pitch often = sadness)
- Speaking rate (fast = excited/angry, slow = sad/calm)
- Volume dynamics (loud = angry, soft = sad/fearful)
- Voice quality (breathy, tense, relaxed)

Common applications:

- Call centers: Detect frustrated customers for priority handling
- Mental health: Monitor patient emotional well-being
- Voice assistants: Respond appropriately to user mood
- Gaming: Adapt gameplay to player emotional state
- Market research: Analyze focus group reactions

Default emotions supported (based on industry standards):

- Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised

You can also measure:

- Arousal: How activated/calm the speaker is (-1 to +1)
- Valence: How positive/negative the emotion is (-1 to +1)

## How It Works

This model uses deep learning to detect emotions from speech audio. It supports two operation modes:

- **ONNX Mode:** Load pre-trained models for fast inference
- **Native Mode:** Train models from scratch with full customization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeechEmotionRecognizer(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double,Int32,Int32,Int32,Double,String[],Boolean,ILossFunction<>)` | Creates a speech emotion recognizer in native training mode. |
| `SpeechEmotionRecognizer(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,String[],Boolean)` | Creates a speech emotion recognizer in ONNX inference mode with a pre-trained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedEmotions` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeArousalFromProbabilities(IReadOnlyDictionary<String,>)` | Computes arousal from already-computed emotion probabilities. |
| `ComputeValenceFromProbabilities(IReadOnlyDictionary<String,>)` | Computes valence from already-computed emotion probabilities. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ExtractEmotionFeatures(Tensor<>)` |  |
| `Forward(Tensor<>)` |  |
| `GetArousal(Tensor<>)` |  |
| `GetEmotionProbabilities(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetValence(Tensor<>)` |  |
| `InitializeLayers` | Initializes the neural network layers for native training mode. |
| `NormalizeMelSpectrogram(Tensor<>)` | Normalizes the mel spectrogram for neural network input. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `RecognizeEmotion(Tensor<>)` |  |
| `RecognizeEmotionTimeSeries(Tensor<>,Int32,Int32)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultEmotions` | Standard emotions supported by this model. |
| `_baseFilters` | Number of filters in the first convolutional layer (doubles with each block). |
| `_convLayers` | Convolutional feature extraction layers. |
| `_denseLayers` | Dense classification layers. |
| `_dropoutRate` | Dropout rate for regularization. |
| `_emotionLabels` | Custom emotion labels if provided. |
| `_hiddenDim` | Hidden dimension for dense layers. |
| `_hopLength` | Hop length between FFT frames. |
| `_includeArousalValence` | Whether to include arousal/valence prediction. |
| `_inputDurationSeconds` | Expected input duration in seconds. |
| `_isOnnxMode` | Indicates whether this model is running in ONNX inference mode. |
| `_melSpec` | Mel spectrogram extractor. |
| `_modelPath` | Path to the ONNX emotion recognition model. |
| `_nFft` | FFT window size for spectrogram computation. |
| `_numConvBlocks` | Number of convolutional blocks in the feature extractor. |
| `_outputLayer` | Output layer for emotion classification. |

