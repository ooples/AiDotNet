---
title: "BEATs<T>"
description: "BEATs (Audio Pre-Training with Acoustic Tokenizers) model for state-of-the-art audio event detection and classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

BEATs (Audio Pre-Training with Acoustic Tokenizers) model for state-of-the-art audio
event detection and classification.

## For Beginners

BEATs is one of the most accurate models for identifying sounds in audio.
It can detect hundreds of different sounds simultaneously - speech, music, animals, machinery,
weather, and more - all from the same audio clip.

Here is how BEATs processes audio, step by step:

**Step 1 - Sound to picture:** Audio waves are converted to a "mel spectrogram" - think
of it as a heat map where the x-axis is time, the y-axis is pitch (frequency), and the
brightness shows how loud each pitch is at each moment. A dog bark would show short bright
bursts in the mid-frequency range, while a whistle would show a thin bright line at high frequency.

**Step 2 - Cut into tiles:** The spectrogram image is cut into small 16x16 pixel tiles
called "patches" (like puzzle pieces). Each patch captures about 0.16 seconds of a narrow
frequency band. A 10-second clip produces roughly 500 patches.

**Step 3 - Describe each tile:** Each small tile is converted into a list of 768 numbers
(a "vector") that describes its contents. This is done by a linear projection layer - essentially
a learned formula that extracts the most important features from each tile.

**Step 4 - Understand context:** This is where the magic happens. A Transformer encoder
(the same architecture behind ChatGPT and modern AI) looks at ALL tiles simultaneously and
figures out how they relate to each other. For example, it learns that:

- A particular frequency pattern repeated rhythmically = drums
- Formant patterns in the 200-4000 Hz range with pauses = speech
- Broadband energy with no harmonic structure = wind or rain

The Transformer has 12 layers, each adding deeper understanding.

**Step 5 - Make predictions:** The Transformer's output is averaged across all tiles and
fed through a classification head that outputs a probability for each of the 527 possible
sounds. Multiple sounds can be detected at once (e.g., "speech: 95%, music: 40%, rain: 15%").

**Why BEATs is special:** Most previous audio models learned by being told what sounds are
in training clips. BEATs instead learns by playing a game with itself: it masks (hides) 75%
of the spectrogram tiles and tries to guess what was hidden. This "self-supervised" approach
means BEATs can learn from millions of unlabeled audio clips, making it much more accurate.

**Usage with a pre-trained ONNX model (recommended for most users):****Usage with native training (for researchers and custom datasets):**

## How It Works

BEATs (Chen et al., ICML 2023) achieves state-of-the-art results on multiple audio benchmarks:

- **AudioSet-2M**: 50.6% mAP (mean Average Precision) - the largest audio benchmark

with 2 million YouTube clips spanning 527 sound event classes

- **ESC-50**: 98.1% accuracy - a dataset of 2000 environmental sound clips

in 50 categories (rain, dog bark, clock tick, etc.)

- **AudioSet-Balanced**: 47.5% mAP - the class-balanced evaluation subset

**Architecture:** BEATs adapts the Vision Transformer (ViT) architecture for audio spectrograms.
The key insight is treating a mel spectrogram like an image and processing it with patches:

- **Audio preprocessing**: Raw waveform (e.g., 16kHz mono) is converted to a 128-bin

log-mel spectrogram, creating a 2D time-frequency representation of the sound

- **Patch embedding**: The spectrogram is divided into non-overlapping 16x16 patches,

and each patch is linearly projected to a 768-dimensional embedding vector

- **Positional encoding**: Learnable positional embeddings are added so the Transformer

knows the spatial ordering of patches (which part of the spectrogram each patch came from)

- **Transformer encoder**: A 12-layer encoder with 12-head self-attention and GELU

feed-forward networks processes the patch sequence, learning relationships between different
time-frequency regions (e.g., a bark onset followed by harmonics = dog barking)

- **Classification head**: The encoded representations are pooled and projected through

a linear layer to produce per-class logits, with sigmoid activation for multi-label detection
(multiple sounds can occur simultaneously in real audio)

**Training Strategy:** BEATs uses a novel iterative self-distillation framework that
alternates between two components:

- **Acoustic Tokenizer**: Converts each audio patch into a discrete token (one of 8192 codes).

This is similar to how text is tokenized into words, but for audio. The tokenizer is trained
using the audio model's representations from the previous iteration.

- **Masked Patch Prediction**: 75% of spectrogram patches are randomly masked, and the model

must predict the acoustic tokens assigned by the tokenizer. This forces the model to learn rich
audio representations by filling in the missing pieces, similar to BERT's masked language modeling.

- **Iteration**: The improved audio model produces better representations, which train a better

tokenizer, which provides better labels for the next round of pre-training. BEATs_iter3 (3 iterations)
achieves the best results.

**References:**

- Paper: "BEATs: Audio Pre-Training with Acoustic Tokenizers" (Chen et al., ICML 2023)
- Repository: https://github.com/microsoft/unilm/tree/master/beats
- AudioSet: https://research.google.com/audioset/ (the primary evaluation benchmark)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BEATs(NeuralNetworkArchitecture<>,BEATsOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BEATs model for native training mode. |
| `BEATs(NeuralNetworkArchitecture<>,String,BEATsOptions)` | Creates a BEATs model for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` | Gets the event labels (alias for `SupportedEvents` for API compatibility). |
| `SupportedEvents` | Gets the list of event types this model can detect. |
| `TimeResolution` | Gets the time resolution for event detection in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClassifyWindow(Tensor<>)` | Classifies a single mel spectrogram window and returns per-class sigmoid probabilities. |
| `ComputeEventStatistics(IReadOnlyList<AudioEvent<>>)` | Computes per-class aggregate statistics from a list of detected events. |
| `CreateAsync(BEATsOptions,IProgress<Double>,CancellationToken)` | Creates a BEATs model asynchronously with optional model download. |
| `CreateNewInstance` | Creates a new BEATs instance for the deserialization framework. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes BEATs-specific model data from a binary stream. |
| `Detect(Tensor<>)` | Detects all audio events in the audio stream using the default confidence threshold. |
| `Detect(Tensor<>,)` | Detects audio events with a custom confidence threshold. |
| `DetectAsync(Tensor<>,CancellationToken)` | Detects audio events asynchronously using a background thread. |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` | Detects only specific event types, filtering out all others. |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` | Detects specific event types with a custom confidence threshold. |
| `Dispose(Boolean)` | Disposes managed and unmanaged resources held by this BEATs instance. |
| `GetEventProbabilities(Tensor<>)` | Gets the raw frame-level event probabilities for all classes without applying a threshold. |
| `GetModelMetadata` | Gets metadata about this BEATs model instance, including architecture details and configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the BEATs neural network layers using either custom user-provided layers or the paper-standard BEATs architecture from `Double)`. |
| `MergeEvents(List<AudioEvent<>>)` | Merges overlapping or adjacent events of the same type into continuous event segments. |
| `PostprocessOutput(Tensor<>)` | Post-processes BEATs model output by applying element-wise sigmoid activation for multi-label classification. |
| `PredictCore(Tensor<>)` | Runs BEATs inference on the given input tensor. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio into a log-mel spectrogram suitable for BEATs input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes all BEATs-specific model data into a binary stream for persistence. |
| `SplitIntoWindows(Tensor<>)` | Splits raw audio into overlapping windows for frame-level BEATs analysis. |
| `StartStreamingSession` | Starts a streaming event detection session with default settings. |
| `StartStreamingSession(Int32,)` | Starts a streaming event detection session with custom sample rate and threshold. |
| `Train(Tensor<>,Tensor<>)` | Trains the BEATs model on a single input-target pair using backpropagation. |
| `UpdateParameters(Vector<>)` | Updates all network parameters from a flattened parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `AudioSetLabels` | AudioSet-527 standard event labels used by BEATs pre-trained models. |

