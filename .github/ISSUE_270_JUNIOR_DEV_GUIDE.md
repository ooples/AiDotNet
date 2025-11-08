# Issue #270: Text-to-Speech (TTS) Implementation Guide

## For Junior Developers: Complete Implementation Tutorial

### Table of Contents
1. [Understanding Text-to-Speech](#understanding-tts)
2. [TTS Pipeline Overview](#tts-pipeline)
3. [FastSpeech 2 Architecture](#fastspeech2-architecture)
4. [HiFi-GAN Vocoder](#hifigan-vocoder)
5. [Implementation Guide](#implementation-guide)
6. [Testing Strategy](#testing-strategy)
7. [Common Pitfalls](#common-pitfalls)

---

## Understanding TTS

### What is Text-to-Speech?

**For Beginners:** Text-to-Speech (TTS) is the technology that converts written text into spoken audio. Think of it as the opposite of ASR (speech recognition). You've heard TTS when:
- GPS navigation gives you directions
- Smart assistants read text messages aloud
- Audiobook narrators (some are now AI-generated)
- Screen readers for accessibility

**How it works at a high level:**
1. **Text Input**: "Hello, how are you today?"
2. **Text Processing**: Clean and tokenize the text
3. **Acoustic Model**: Generate mel-spectrogram (FastSpeech 2)
4. **Vocoder**: Convert spectrogram to audio waveform (HiFi-GAN)
5. **Audio Output**: Raw audio you can play through speakers

**Real-world analogy:**
- **FastSpeech 2** is like a musician reading sheet music and deciding how to play each note (timing, pitch, emphasis)
- **HiFi-GAN** is like the instrument that actually produces the sound based on those instructions

**Key difference from ASR:**
- **ASR**: Audio → Text (listening)
- **TTS**: Text → Audio (speaking)

---

## TTS Pipeline

### Two-Stage Architecture

Modern TTS uses a **two-stage pipeline**:

```
Stage 1: Text → Mel-Spectrogram (Acoustic Model)
         ↓
    FastSpeech 2
         ↓
  Mel-Spectrogram (visual representation of sound)

Stage 2: Mel-Spectrogram → Audio Waveform (Vocoder)
         ↓
     HiFi-GAN
         ↓
  Raw Audio (playable sound)
```

**Why two stages?**

1. **Separation of concerns**:
   - Stage 1: Figure out WHAT to say (phonemes, timing, prosody)
   - Stage 2: Figure out HOW to say it (voice quality, naturalness)

2. **Training efficiency**:
   - Mel-spectrograms are much smaller than raw audio
   - Easier to train and faster to generate

3. **Flexibility**:
   - Can swap different acoustic models (FastSpeech, Tacotron)
   - Can swap different vocoders (HiFi-GAN, WaveGlow, WaveNet)

### What is Prosody?

**For Beginners:** Prosody is the rhythm, stress, and intonation of speech. It's what makes speech sound natural vs robotic.

**Examples:**
- **Stress**: "I didn't say YOU stole it" (someone else did)
- **Stress**: "I didn't say you STOLE it" (you borrowed it)
- **Intonation**: "You're going?" (question) vs "You're going." (statement)
- **Rhythm**: Pauses between words, speed of speaking

FastSpeech 2 learns to predict prosody from text context.

---

## FastSpeech 2 Architecture

### Overview

FastSpeech 2 is a **non-autoregressive** acoustic model that generates mel-spectrograms from text **in parallel** (much faster than autoregressive models like Tacotron).

**Key features:**
- **Feed-forward architecture**: Processes all text at once
- **Duration prediction**: Learns how long each phoneme should be
- **Variance adaptors**: Predicts pitch and energy for naturalness
- **Fast inference**: Generates spectrograms 50x faster than Tacotron 2

### Architecture Components

```
Input: Text Tokens [H, e, l, l, o]
  ↓
Phoneme Embedding + Positional Encoding
  ↓
Feed-Forward Transformer (Encoder)
  - Multi-Head Self-Attention
  - 1D Convolution layers
  - Layer Normalization
  ↓
Variance Adaptors (in parallel):
  ├─ Duration Predictor → How long each phoneme
  ├─ Pitch Predictor → F0 contour (voice pitch)
  └─ Energy Predictor → Volume/emphasis
  ↓
Length Regulator (expand based on durations)
  ↓
Feed-Forward Transformer (Decoder)
  ↓
Linear Projection to Mel Channels (80)
  ↓
Output: Mel-Spectrogram (time_steps, 80)
```

### 1. Text Preprocessing

**For Beginners:** Before feeding text to the model, we need to convert it to a format the model understands.

**Steps:**

a) **Text Normalization**
```csharp
// Convert to lowercase
string normalized = text.ToLowerInvariant();

// Expand abbreviations
normalized = normalized.Replace("dr.", "doctor");
normalized = normalized.Replace("mr.", "mister");

// Handle numbers
normalized = normalized.Replace("123", "one hundred twenty three");
```

b) **Phoneme Conversion** (Optional but recommended)
```
Text: "Hello world"
Phonemes: [HH, EH, L, OW, W, ER, L, D]
```

**Why phonemes?** Same spelling can have different pronunciations:
- "read" (present): [R, IY, D]
- "read" (past): [R, EH, D]

c) **Tokenization**
```csharp
// Map each character/phoneme to an integer ID
string text = "hello";
int[] tokens = { 8, 5, 12, 12, 15 }; // Map h→8, e→5, l→12, o→15
```

### 2. Duration Prediction

**For Beginners:** Different sounds take different amounts of time. "S" is quick, but "aaa" can be stretched.

**How it works:**
- Model predicts duration (in frames) for each phoneme
- Duration = number of mel-spectrogram frames to generate for that phoneme

**Example:**
```
Text:     "H  e  l  l  o"
Durations: 3  4  2  2  5  (frames per phoneme)
Total:    16 frames in output spectrogram
```

**Implementation:**
```csharp
// Duration predictor is a small CNN
// Input: Encoder hidden states (text features)
// Output: Duration for each phoneme
Vector<T> durations = DurationPredictor.Forward(encoderOutput);

// Durations must be positive integers
for (int i = 0; i < durations.Length; i++)
{
    durations[i] = NumOps.Max(durations[i], NumOps.One);
}
```

### 3. Variance Adaptors

**Pitch Predictor:**
- Predicts fundamental frequency (F0) - how high/low the voice is
- Female voices: ~200 Hz, Male voices: ~120 Hz
- Varies over time for natural intonation

**Energy Predictor:**
- Predicts volume/loudness for each frame
- Helps with emphasis and stress patterns

**For Beginners:** These predictors add expressiveness. Without them, speech sounds monotone (same pitch, same volume).

### 4. Length Regulator

**For Beginners:** This expands the phoneme sequence to match predicted durations.

**Example:**
```
Input:     [h, e, l, l, o]     (5 phonemes)
Durations: [3, 4, 2, 2, 5]     (frames per phoneme)
Output:    [h, h, h, e, e, e, e, l, l, l, l, o, o, o, o, o]  (16 frames)
```

**Implementation:**
```csharp
// Repeat each phoneme embedding by its duration
Vector<T> expandedSequence = new Vector<T>(totalFrames * hiddenDim);
int outputIdx = 0;

for (int i = 0; i < phonemes.Length; i++)
{
    int duration = (int)durations[i];
    for (int d = 0; d < duration; d++)
    {
        // Copy phoneme embedding
        for (int j = 0; j < hiddenDim; j++)
        {
            expandedSequence[outputIdx * hiddenDim + j] = encoderOutput[i * hiddenDim + j];
        }
        outputIdx++;
    }
}
```

### 5. Decoder

**For Beginners:** The decoder refines the expanded sequence into a smooth mel-spectrogram.

- Similar architecture to encoder (transformer blocks)
- Input: Expanded phoneme features + pitch + energy
- Output: Mel-spectrogram (frames, 80 mel channels)

---

## HiFi-GAN Vocoder

### Overview

HiFi-GAN is a **Generative Adversarial Network (GAN)** that converts mel-spectrograms into high-quality audio waveforms.

**For Beginners:** Think of HiFi-GAN as a super-advanced "upscaler" that turns a low-resolution image (spectrogram) into a high-resolution audio (waveform).

**Key innovation:** Uses multiple discriminators at different scales to produce high-fidelity audio (up to 22kHz or 24kHz quality).

### Architecture

```
Input: Mel-Spectrogram (time, 80)
  ↓
Upsampling Layers (Transposed Convolutions)
  - Upsample by 8x → 4x → 2x → 2x (total 256x)
  - Each layer doubles/quadruples time resolution
  ↓
Residual Blocks with Dilated Convolutions
  - Capture patterns at multiple time scales
  - Dilations: 1, 3, 5 (like looking at different zoom levels)
  ↓
Final Convolution → Tanh Activation
  ↓
Output: Raw Audio Waveform (-1.0 to 1.0)
```

### Upsampling Process

**For Beginners:** The mel-spectrogram has ~100 frames per second, but audio at 22kHz has 22,000 samples per second. We need to upsample by ~220x.

**Example upsampling:**
```
Input:  Mel-spec at 100 Hz     (100 frames/sec)
        ↓ Upsample 8x
        800 Hz
        ↓ Upsample 4x
        3,200 Hz
        ↓ Upsample 2x
        6,400 Hz
        ↓ Upsample 2x
Output: 12,800 Hz (could continue to 22,050 Hz)
```

**Transposed convolution:**
- Regular convolution: Reduces size (downsampling)
- Transposed convolution: Increases size (upsampling)
- Learns how to fill in missing samples intelligently

### Multi-Scale Discriminators

**For Beginners:** During training (not inference), multiple "critics" check if the generated audio sounds real at different scales:

1. **High-frequency discriminator**: Checks fine details (consonants, sibilants)
2. **Mid-frequency discriminator**: Checks vowels, pitch
3. **Low-frequency discriminator**: Checks rhythm, prosody

This ensures the generator produces realistic audio at all frequencies.

**Note:** For inference (our use case), we only use the generator - discriminators are training-only.

---

## Implementation Guide

### Phase 1: Text Preprocessor

**File:** `src/Audio/TextPreprocessor.cs`

```csharp
using AiDotNet.Helpers;
using System.Text.RegularExpressions;

namespace AiDotNet.Audio
{
    /// <summary>
    /// Preprocesses text for TTS by normalizing and tokenizing.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This class converts messy text ("Dr. Smith has 3 cats!")
    /// into a clean format the model understands ([D, R, space, S, M, I, T, H, ...]).
    /// </remarks>
    public class TextPreprocessor
    {
        private readonly Dictionary<char, int> _charToId;
        private readonly Dictionary<int, char> _idToChar;

        /// <summary>
        /// Creates a text preprocessor with a character vocabulary.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> We need a mapping between characters and numbers.
        /// For example: 'a'→0, 'b'→1, 'c'→2, etc.
        /// The model works with numbers, not letters.
        /// </remarks>
        public TextPreprocessor()
        {
            // Build vocabulary: a-z, space, punctuation
            string vocab = " abcdefghijklmnopqrstuvwxyz.,!?'-";
            _charToId = new Dictionary<char, int>();
            _idToChar = new Dictionary<int, char>();

            for (int i = 0; i < vocab.Length; i++)
            {
                _charToId[vocab[i]] = i;
                _idToChar[i] = vocab[i];
            }
        }

        /// <summary>
        /// Normalizes and tokenizes text into token IDs.
        /// </summary>
        /// <param name="text">Input text to convert.</param>
        /// <returns>Array of token IDs.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This performs several steps:
        /// 1. Lowercase: "Hello" → "hello"
        /// 2. Expand abbreviations: "Dr." → "doctor"
        /// 3. Remove unsupported characters
        /// 4. Convert to token IDs: "hi" → [8, 9]
        /// </remarks>
        public int[] Tokenize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return Array.Empty<int>();

            // Step 1: Normalize
            string normalized = Normalize(text);

            // Step 2: Convert to token IDs
            var tokens = new List<int>();
            foreach (char c in normalized)
            {
                if (_charToId.TryGetValue(c, out int id))
                {
                    tokens.Add(id);
                }
                // Skip unsupported characters
            }

            return tokens.ToArray();
        }

        /// <summary>
        /// Converts token IDs back to text.
        /// </summary>
        public string Detokenize(int[] tokens)
        {
            var chars = new char[tokens.Length];
            for (int i = 0; i < tokens.Length; i++)
            {
                if (_idToChar.TryGetValue(tokens[i], out char c))
                {
                    chars[i] = c;
                }
                else
                {
                    chars[i] = '?'; // Unknown token
                }
            }
            return new string(chars);
        }

        /// <summary>
        /// Normalizes text by cleaning and expanding abbreviations.
        /// </summary>
        private string Normalize(string text)
        {
            // Lowercase
            text = text.ToLowerInvariant();

            // Expand common abbreviations
            text = text.Replace("dr.", "doctor");
            text = text.Replace("mr.", "mister");
            text = text.Replace("mrs.", "misses");
            text = text.Replace("ms.", "miss");
            text = text.Replace("st.", "street");
            text = text.Replace("ave.", "avenue");

            // Convert numbers to words (simplified - production would use full number converter)
            text = Regex.Replace(text, @"\b0\b", "zero");
            text = Regex.Replace(text, @"\b1\b", "one");
            text = Regex.Replace(text, @"\b2\b", "two");
            text = Regex.Replace(text, @"\b3\b", "three");
            text = Regex.Replace(text, @"\b4\b", "four");
            text = Regex.Replace(text, @"\b5\b", "five");
            text = Regex.Replace(text, @"\b6\b", "six");
            text = Regex.Replace(text, @"\b7\b", "seven");
            text = Regex.Replace(text, @"\b8\b", "eight");
            text = Regex.Replace(text, @"\b9\b", "nine");

            // Remove multiple spaces
            text = Regex.Replace(text, @"\s+", " ");

            return text.Trim();
        }

        /// <summary>
        /// Gets the vocabulary size.
        /// </summary>
        public int VocabSize => _charToId.Count;
    }
}
```

### Phase 2: TtsModel Wrapper

**File:** `src/Models/Audio/TtsModel.cs`

```csharp
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Audio
{
    /// <summary>
    /// Wrapper for FastSpeech 2 + HiFi-GAN ONNX models to perform text-to-speech synthesis.
    /// </summary>
    /// <typeparam name="T">Numeric type for computations.</typeparam>
    /// <remarks>
    /// <b>For Beginners:</b> This class orchestrates the two-stage TTS pipeline:
    /// 1. FastSpeech 2: Text → Mel-Spectrogram (what to say)
    /// 2. HiFi-GAN: Mel-Spectrogram → Audio Waveform (how to say it)
    ///
    /// Think of it like a production pipeline:
    /// - Stage 1: Composer writes the musical score
    /// - Stage 2: Orchestra performs the score
    /// </remarks>
    public class TtsModel<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        private readonly OnnxModel<T> _fastSpeech2Model;
        private readonly OnnxModel<T> _vocoderModel;
        private readonly TextPreprocessor _preprocessor;

        private readonly int _sampleRate;

        /// <summary>
        /// Creates a new TTS model instance.
        /// </summary>
        /// <param name="fastSpeech2Path">Path to FastSpeech 2 ONNX model.</param>
        /// <param name="vocoderPath">Path to HiFi-GAN vocoder ONNX model.</param>
        /// <param name="sampleRate">Audio sample rate (default: 22050 Hz).</param>
        /// <remarks>
        /// <b>For Beginners:</b> You need TWO model files:
        /// 1. FastSpeech 2 ONNX (acoustic model)
        /// 2. HiFi-GAN ONNX (vocoder)
        ///
        /// Download from Hugging Face or convert PyTorch models.
        ///
        /// <b>Sample rate:</b> 22050 Hz is standard for TTS (good quality, efficient).
        /// Higher rates (44.1kHz) produce better quality but slower generation.
        /// </remarks>
        public TtsModel(
            string fastSpeech2Path,
            string vocoderPath,
            int sampleRate = 22050)
        {
            if (string.IsNullOrWhiteSpace(fastSpeech2Path))
                throw new ArgumentException("FastSpeech 2 path cannot be empty", nameof(fastSpeech2Path));
            if (string.IsNullOrWhiteSpace(vocoderPath))
                throw new ArgumentException("Vocoder path cannot be empty", nameof(vocoderPath));
            if (!File.Exists(fastSpeech2Path))
                throw new FileNotFoundException($"FastSpeech 2 model not found: {fastSpeech2Path}");
            if (!File.Exists(vocoderPath))
                throw new FileNotFoundException($"Vocoder model not found: {vocoderPath}");
            if (sampleRate <= 0)
                throw new ArgumentException("Sample rate must be positive", nameof(sampleRate));

            _fastSpeech2Model = new OnnxModel<T>(fastSpeech2Path);
            _vocoderModel = new OnnxModel<T>(vocoderPath);
            _preprocessor = new TextPreprocessor();
            _sampleRate = sampleRate;
        }

        /// <summary>
        /// Synthesizes speech from text.
        /// </summary>
        /// <param name="text">Text to convert to speech.</param>
        /// <param name="speakingRate">Speed multiplier (default: 1.0 = normal speed).</param>
        /// <returns>Audio waveform tensor.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is the main method you'll call. Pass in text,
        /// get back raw audio samples you can save to a WAV file or play.
        ///
        /// <b>Example:</b>
        /// <code>
        /// var tts = new TtsModel&lt;float&gt;("fastspeech2.onnx", "hifigan.onnx");
        /// Tensor&lt;float&gt; audio = tts.Synthesize("Hello, world!");
        /// SaveWavFile("hello.wav", audio, sampleRate: 22050);
        /// </code>
        ///
        /// <b>Speaking rate:</b>
        /// - 0.5 = half speed (slower, clearer)
        /// - 1.0 = normal speed
        /// - 1.5 = 1.5x speed (faster)
        /// </remarks>
        public Tensor<T> Synthesize(string text, double speakingRate = 1.0)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be empty", nameof(text));
            if (speakingRate <= 0.0)
                throw new ArgumentException("Speaking rate must be positive", nameof(speakingRate));

            // Step 1: Preprocess text to token IDs
            int[] tokens = _preprocessor.Tokenize(text);

            if (tokens.Length == 0)
                throw new ArgumentException("Text contains no valid characters");

            // Step 2: Generate mel-spectrogram with FastSpeech 2
            Matrix<T> melSpectrogram = GenerateMelSpectrogram(tokens, speakingRate);

            // Step 3: Generate audio waveform with HiFi-GAN
            Tensor<T> audioWaveform = GenerateAudio(melSpectrogram);

            return audioWaveform;
        }

        /// <summary>
        /// Generates mel-spectrogram from token IDs using FastSpeech 2.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is Stage 1 of the pipeline.
        /// FastSpeech 2 takes text tokens and predicts:
        /// - Duration for each phoneme (how long to say it)
        /// - Pitch contour (voice intonation)
        /// - Energy (loudness/emphasis)
        /// - Final mel-spectrogram (visual representation of sound)
        /// </remarks>
        private Matrix<T> GenerateMelSpectrogram(int[] tokens, double speakingRate)
        {
            // Convert tokens to tensor
            Tensor<T> tokenTensor = TokensToTensor(tokens);

            // Prepare inputs for FastSpeech 2
            var inputs = new Dictionary<string, Tensor<T>>
            {
                { "tokens", tokenTensor }
            };

            // Optional: Control speaking rate via duration scaling
            if (Math.Abs(speakingRate - 1.0) > 0.01)
            {
                var speedTensor = new Tensor<T>(new int[] { 1 });
                speedTensor[0] = NumOps.FromDouble(1.0 / speakingRate); // Inverse for duration
                inputs["speed"] = speedTensor;
            }

            // Run FastSpeech 2 forward pass
            var outputs = _fastSpeech2Model.Forward(inputs);

            // Extract mel-spectrogram
            Tensor<T> melTensor = outputs["mel"]; // Output name may vary

            // Convert to matrix (time_steps, n_mels)
            return TensorToMatrix(melTensor);
        }

        /// <summary>
        /// Generates audio waveform from mel-spectrogram using HiFi-GAN.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is Stage 2 of the pipeline.
        /// HiFi-GAN "renders" the mel-spectrogram into actual audio:
        /// - Upsamples time resolution (100 Hz → 22050 Hz)
        /// - Adds fine acoustic details (harmonics, formants)
        /// - Produces natural-sounding voice
        ///
        /// Think of mel-spectrogram as a blueprint, and HiFi-GAN as the builder
        /// constructing the final product (audio waveform).
        /// </remarks>
        private Tensor<T> GenerateAudio(Matrix<T> melSpectrogram)
        {
            // Convert mel-spectrogram to tensor
            Tensor<T> melTensor = MatrixToTensor(melSpectrogram);

            // Prepare inputs for vocoder
            var inputs = new Dictionary<string, Tensor<T>>
            {
                { "mel", melTensor }
            };

            // Run HiFi-GAN forward pass
            var outputs = _vocoderModel.Forward(inputs);

            // Extract audio waveform
            Tensor<T> audioTensor = outputs["audio"]; // Output name may vary

            return audioTensor;
        }

        /// <summary>
        /// Converts token IDs to tensor for ONNX input.
        /// </summary>
        private Tensor<T> TokensToTensor(int[] tokens)
        {
            int[] shape = new int[] { 1, tokens.Length }; // (batch=1, sequence_length)
            var data = new Vector<T>(tokens.Length);

            for (int i = 0; i < tokens.Length; i++)
            {
                data[i] = NumOps.FromDouble(tokens[i]);
            }

            return new Tensor<T>(shape, data);
        }

        /// <summary>
        /// Converts ONNX output tensor to matrix.
        /// </summary>
        private Matrix<T> TensorToMatrix(Tensor<T> tensor)
        {
            // Expected shape: (batch=1, time_steps, n_mels) or (time_steps, n_mels)
            int[] shape = tensor.Shape;

            int timeSteps, nMels;
            if (shape.Length == 3)
            {
                // (batch, time, mels)
                timeSteps = shape[1];
                nMels = shape[2];
            }
            else if (shape.Length == 2)
            {
                // (time, mels)
                timeSteps = shape[0];
                nMels = shape[1];
            }
            else
            {
                throw new ArgumentException($"Unexpected tensor shape: {string.Join(", ", shape)}");
            }

            var matrix = new Matrix<T>(timeSteps, nMels);

            for (int t = 0; t < timeSteps; t++)
            {
                for (int m = 0; m < nMels; m++)
                {
                    matrix[t, m] = shape.Length == 3
                        ? tensor[0, t, m]
                        : tensor[t, m];
                }
            }

            return matrix;
        }

        /// <summary>
        /// Converts matrix to tensor for ONNX input.
        /// </summary>
        private Tensor<T> MatrixToTensor(Matrix<T> matrix)
        {
            // HiFi-GAN typically expects: (batch=1, n_mels, time_steps)
            // Note: Different from FastSpeech 2 which uses (batch, time, mels)

            int[] shape = new int[] { 1, matrix.Columns, matrix.Rows };
            var data = new Vector<T>(matrix.Rows * matrix.Columns);

            int idx = 0;
            // Transpose: (time, mels) → (mels, time)
            for (int m = 0; m < matrix.Columns; m++)
            {
                for (int t = 0; t < matrix.Rows; t++)
                {
                    data[idx++] = matrix[t, m];
                }
            }

            return new Tensor<T>(shape, data);
        }

        /// <summary>
        /// Gets the audio sample rate.
        /// </summary>
        public int SampleRate => _sampleRate;
    }
}
```

### Phase 3: Audio File Helper

**File:** `src/Audio/AudioFileHelper.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using System.IO;

namespace AiDotNet.Audio
{
    /// <summary>
    /// Utilities for saving audio to WAV files.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> WAV is a simple, uncompressed audio format.
    /// It stores raw audio samples with a small header describing the format.
    /// Perfect for testing TTS output before converting to MP3/AAC.
    /// </remarks>
    public static class AudioFileHelper
    {
        /// <summary>
        /// Saves audio tensor to WAV file.
        /// </summary>
        /// <typeparam name="T">Numeric type.</typeparam>
        /// <param name="audio">Audio waveform (-1.0 to 1.0 range).</param>
        /// <param name="filePath">Output file path (.wav).</param>
        /// <param name="sampleRate">Sample rate in Hz (e.g., 22050).</param>
        /// <param name="numChannels">Number of channels (1=mono, 2=stereo).</param>
        /// <remarks>
        /// <b>For Beginners:</b> This converts the raw numbers from TTS into a
        /// playable audio file you can open in any audio player.
        ///
        /// <b>Format details:</b>
        /// - 16-bit PCM: Each sample is a 16-bit integer (-32768 to 32767)
        /// - Mono: Single channel (typical for TTS)
        /// - Sample rate: How many samples per second (22050 = 22.05 kHz)
        /// </remarks>
        public static void SaveWav<T>(
            Tensor<T> audio,
            string filePath,
            int sampleRate,
            int numChannels = 1)
        {
            var numOps = MathHelper.GetNumericOperations<T>();

            if (audio == null)
                throw new ArgumentNullException(nameof(audio));
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path cannot be empty", nameof(filePath));
            if (sampleRate <= 0)
                throw new ArgumentException("Sample rate must be positive", nameof(sampleRate));
            if (numChannels < 1 || numChannels > 2)
                throw new ArgumentException("Channels must be 1 (mono) or 2 (stereo)", nameof(numChannels));

            // Convert audio to flat array
            int numSamples = audio.Shape[audio.Shape.Length - 1];
            var samples = new short[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                // Get sample value
                T value = audio.Shape.Length == 1 ? audio[i] : audio[0, i];

                // Convert to double
                double sampleValue = Convert.ToDouble(value);

                // Clamp to [-1.0, 1.0]
                sampleValue = Math.Max(-1.0, Math.Min(1.0, sampleValue));

                // Convert to 16-bit PCM
                samples[i] = (short)(sampleValue * 32767.0);
            }

            // Write WAV file
            using (var writer = new BinaryWriter(File.Create(filePath)))
            {
                // WAV header (44 bytes)
                int byteRate = sampleRate * numChannels * 2; // 2 = 16-bit
                int dataSize = numSamples * numChannels * 2;

                // RIFF header
                writer.Write(new char[] { 'R', 'I', 'F', 'F' });
                writer.Write(36 + dataSize); // File size - 8
                writer.Write(new char[] { 'W', 'A', 'V', 'E' });

                // fmt chunk
                writer.Write(new char[] { 'f', 'm', 't', ' ' });
                writer.Write(16); // Chunk size
                writer.Write((short)1); // Audio format (1 = PCM)
                writer.Write((short)numChannels);
                writer.Write(sampleRate);
                writer.Write(byteRate);
                writer.Write((short)(numChannels * 2)); // Block align
                writer.Write((short)16); // Bits per sample

                // data chunk
                writer.Write(new char[] { 'd', 'a', 't', 'a' });
                writer.Write(dataSize);

                // Write audio data
                foreach (short sample in samples)
                {
                    writer.Write(sample);
                }
            }
        }
    }
}
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/UnitTests/Models/TtsModelTests.cs`

```csharp
using Xunit;
using Moq;
using AiDotNet.Models.Audio;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Models
{
    public class TtsModelTests
    {
        [Fact]
        public void Synthesize_CallsFastSpeech2ThenVocoder()
        {
            // Arrange
            var mockFastSpeech2 = new Mock<OnnxModel<float>>();
            var mockVocoder = new Mock<OnnxModel<float>>();

            // Mock FastSpeech 2 output (mel-spectrogram)
            var melSpec = new Tensor<float>(new int[] { 1, 100, 80 }); // (batch, time, mels)
            mockFastSpeech2.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns(new Dictionary<string, Tensor<float>>
                {
                    { "mel", melSpec }
                });

            // Mock vocoder output (audio waveform)
            var audioWave = new Tensor<float>(new int[] { 1, 22050 }); // 1 second at 22kHz
            mockVocoder.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns(new Dictionary<string, Tensor<float>>
                {
                    { "audio", audioWave }
                });

            var tts = new TtsModel<float>(mockFastSpeech2.Object, mockVocoder.Object);

            // Act
            var result = tts.Synthesize("hello world");

            // Assert
            mockFastSpeech2.Verify(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()), Times.Once);
            mockVocoder.Verify(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()), Times.Once);
            Assert.NotNull(result);
        }

        [Fact]
        public void Synthesize_EmptyText_ThrowsException()
        {
            // Arrange
            var tts = new TtsModel<float>("dummy1.onnx", "dummy2.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => tts.Synthesize(""));
            Assert.Throws<ArgumentException>(() => tts.Synthesize("   "));
        }

        [Theory]
        [InlineData("hello")]
        [InlineData("Hello, world!")]
        [InlineData("The quick brown fox.")]
        public void Synthesize_ValidText_ProducesNonEmptyOutput(string text)
        {
            // Arrange
            var mockFastSpeech2 = new Mock<OnnxModel<float>>();
            var mockVocoder = new Mock<OnnxModel<float>>();

            // Setup mocks to return valid tensors
            var melSpec = new Tensor<float>(new int[] { 1, 100, 80 });
            mockFastSpeech2.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns(new Dictionary<string, Tensor<float>> { { "mel", melSpec } });

            var audioWave = new Tensor<float>(new int[] { 1, 22050 });
            mockVocoder.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns(new Dictionary<string, Tensor<float>> { { "audio", audioWave } });

            var tts = new TtsModel<float>(mockFastSpeech2.Object, mockVocoder.Object);

            // Act
            var result = tts.Synthesize(text);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Shape[result.Shape.Length - 1] > 0);
        }
    }
}
```

### Integration Tests

**File:** `tests/IntegrationTests/Models/TtsModelIntegrationTests.cs`

```csharp
using Xunit;
using AiDotNet.Models.Audio;
using AiDotNet.Audio;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.Models
{
    public class TtsModelIntegrationTests
    {
        [Fact(Skip = "Requires FastSpeech 2 and HiFi-GAN ONNX models")]
        public void Synthesize_RealModels_GeneratesAudio()
        {
            // Arrange
            string fastSpeech2Path = "fastspeech2.onnx";
            string vocoderPath = "hifigan.onnx";

            if (!File.Exists(fastSpeech2Path) || !File.Exists(vocoderPath))
            {
                Assert.True(false, "Download models from Hugging Face or convert PyTorch models");
            }

            var tts = new TtsModel<float>(fastSpeech2Path, vocoderPath, sampleRate: 22050);

            // Act
            Tensor<float> audio = tts.Synthesize("Hello, this is a test of text to speech.");

            // Assert
            Assert.NotNull(audio);
            Assert.True(audio.Shape[audio.Shape.Length - 1] > 10000); // At least 0.5 seconds

            // Save for manual verification
            string outputPath = "tts_output_test.wav";
            AudioFileHelper.SaveWav(audio, outputPath, sampleRate: 22050);

            Assert.True(File.Exists(outputPath));
            Console.WriteLine($"Audio saved to: {outputPath}");
            Console.WriteLine("Play the file to verify speech quality.");
        }

        [Fact(Skip = "Requires models")]
        public void Synthesize_DifferentSpeakingRates_ProducesDifferentDurations()
        {
            // Arrange
            var tts = new TtsModel<float>("fastspeech2.onnx", "hifigan.onnx");
            string text = "The quick brown fox jumps over the lazy dog.";

            // Act
            var normalSpeed = tts.Synthesize(text, speakingRate: 1.0);
            var fastSpeed = tts.Synthesize(text, speakingRate: 1.5);
            var slowSpeed = tts.Synthesize(text, speakingRate: 0.7);

            // Assert
            int normalSamples = normalSpeed.Shape[normalSpeed.Shape.Length - 1];
            int fastSamples = fastSpeed.Shape[fastSpeed.Shape.Length - 1];
            int slowSamples = slowSpeed.Shape[slowSpeed.Shape.Length - 1];

            Assert.True(fastSamples < normalSamples, "Fast speech should be shorter");
            Assert.True(slowSamples > normalSamples, "Slow speech should be longer");

            // Save all for comparison
            AudioFileHelper.SaveWav(normalSpeed, "tts_normal.wav", 22050);
            AudioFileHelper.SaveWav(fastSpeed, "tts_fast.wav", 22050);
            AudioFileHelper.SaveWav(slowSpeed, "tts_slow.wav", 22050);
        }
    }
}
```

---

## Common Pitfalls

### 1. Text Preprocessing Issues

**Problem:** Failing to normalize numbers/abbreviations
```csharp
// ❌ Wrong: Model doesn't know how to pronounce "123"
tts.Synthesize("I have 123 apples");

// ✅ Correct: Expand to words
TextPreprocessor normalizes: "I have one hundred twenty three apples"
```

### 2. Mel-Spectrogram Dimension Mismatch

**Problem:** FastSpeech 2 expects (batch, time, mels), HiFi-GAN expects (batch, mels, time)

**Solution:** Transpose when passing to vocoder
```csharp
// FastSpeech 2 output: (1, 100, 80)
// HiFi-GAN expects: (1, 80, 100)
Matrix<T> transposed = TransposeMatrix(melSpec);
```

### 3. Audio Clipping

**Problem:** Generated audio exceeds [-1.0, 1.0] range, causing distortion

**Solution:** Normalize before saving
```csharp
T maxAbs = NumOps.Zero;
for (int i = 0; i < audio.Length; i++)
{
    T absValue = NumOps.Abs(audio[i]);
    if (NumOps.GreaterThan(absValue, maxAbs))
        maxAbs = absValue;
}

// Normalize if exceeds range
if (NumOps.GreaterThan(maxAbs, NumOps.One))
{
    for (int i = 0; i < audio.Length; i++)
    {
        audio[i] = NumOps.Divide(audio[i], maxAbs);
    }
}
```

### 4. Sample Rate Mismatch

**Problem:** Model trained at 22kHz, but saving at 16kHz (audio sounds sped up)

**Solution:** Use consistent sample rate
```csharp
// Model was trained at 22050 Hz
var tts = new TtsModel<float>(fs2Path, vocoderPath, sampleRate: 22050);
AudioFileHelper.SaveWav(audio, "output.wav", sampleRate: 22050); // Match!
```

### 5. Duration Control Confusion

**Problem:** Speaking rate parameter doesn't work as expected

**Solution:** Understand the relationship
```csharp
// speakingRate = 1.5 → 1.5x faster → SHORTER duration
// duration_scale = 1.0 / speakingRate = 0.667
// Each phoneme lasts 0.667x as long → faster speech
```

### 6. Out-of-Vocabulary Characters

**Problem:** Input contains characters not in vocabulary (emoji, special symbols)

**Solution:** Filter or replace
```csharp
// Remove unsupported characters during tokenization
public int[] Tokenize(string text)
{
    var tokens = new List<int>();
    foreach (char c in text)
    {
        if (_charToId.ContainsKey(c))
            tokens.Add(_charToId[c]);
        // Else: skip silently or replace with space
    }
    return tokens.ToArray();
}
```

---

## Resources

### Papers
- **FastSpeech 2**: "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (Microsoft, 2021)
- **HiFi-GAN**: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)
- **Tacotron 2**: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Google, 2017)

### Code Examples
- FastSpeech 2 (PyTorch): https://github.com/ming024/FastSpeech2
- HiFi-GAN (official): https://github.com/jik876/hifi-gan
- Microsoft Speech SDK: https://github.com/microsoft/cognitive-services-speech-sdk

### ONNX Models
- Hugging Face TTS models: https://huggingface.co/models?pipeline_tag=text-to-speech
- ONNX model zoo: https://github.com/onnx/models

### Audio Processing
- Understanding mel-spectrograms: https://medium.com/analytics-vidhya/understanding-mel-spectrograms-706c9d2e56de
- Prosody in TTS: https://en.wikipedia.org/wiki/Prosody_(linguistics)

---

## Next Steps

After implementing Issue #270, you'll have:
1. Complete text-to-speech pipeline
2. Understanding of acoustic modeling and vocoding
3. Foundation for voice cloning and multi-speaker TTS

You can extend this to:
- Multi-speaker TTS (add speaker embeddings)
- Emotion control (angry, happy, sad prosody)
- Voice cloning (fine-tune on specific speaker)
- Real-time streaming TTS

Good luck with your implementation!
