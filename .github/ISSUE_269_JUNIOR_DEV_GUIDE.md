# Issue #269: Automatic Speech Recognition (ASR) Implementation Guide

## For Junior Developers: Complete Implementation Tutorial

### Table of Contents
1. [Understanding Automatic Speech Recognition](#understanding-asr)
2. [Audio Processing Fundamentals](#audio-fundamentals)
3. [Whisper Architecture Deep Dive](#whisper-architecture)
4. [Implementation Guide](#implementation-guide)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)

---

## Understanding ASR

### What is Automatic Speech Recognition?

**For Beginners:** Automatic Speech Recognition (ASR) is the technology that converts spoken language (audio) into written text. You've used this every time you talk to Siri, Alexa, or use voice typing on your phone.

**How it works at a high level:**
1. **Audio Input**: Record someone speaking (raw audio waveform)
2. **Feature Extraction**: Convert the waveform into a format the model understands (mel-spectrogram)
3. **Encoder**: Process the audio features to understand what sounds are present
4. **Decoder**: Generate text that matches those sounds
5. **Text Output**: Return the transcribed text

**Real-world analogy:** Think of ASR like a skilled stenographer who listens to speech and types what they hear. The encoder is like their ears understanding the sounds, and the decoder is like their brain converting those sounds into written words.

---

## Audio Fundamentals

### 1. Understanding Audio Waveforms

**What is a waveform?**
- Audio is a vibration traveling through air (sound waves)
- When we record audio, we measure air pressure thousands of times per second
- Each measurement is a number representing how much the air pressure changed
- These numbers form a waveform - a 1D array of values over time

**Example:**
```csharp
// A simple 1-second audio clip at 16kHz sampling rate
// Contains 16,000 samples (measurements)
Vector<float> audioWaveform = new Vector<float>(16000);
// Values typically range from -1.0 to 1.0
```

### 2. Sampling Rate

**For Beginners:** The sampling rate determines how many times per second we measure the audio.

- **8 kHz**: Telephone quality (bare minimum for speech)
- **16 kHz**: Standard for speech recognition (Whisper uses this)
- **44.1 kHz**: CD quality (music)
- **48 kHz**: Professional audio

**Why 16 kHz for speech?**
Human speech frequencies are mostly below 8 kHz. By the Nyquist theorem, we need to sample at least 2x the highest frequency, so 16 kHz is perfect for speech.

### 3. What is a Spectrogram?

**For Beginners:** A spectrogram is a visual representation of audio that shows:
- **X-axis**: Time (when sounds occur)
- **Y-axis**: Frequency (what pitch/tone)
- **Color/Intensity**: How loud each frequency is

**Think of it like this:** If a waveform is like reading music left-to-right over time, a spectrogram is like sheet music showing all the notes vertically.

**Creating a spectrogram:**
```
Waveform → Short-Time Fourier Transform (STFT) → Spectrogram
```

### 4. Mel-Spectrogram (Critical for ASR)

**What makes it special?**
The "mel" scale converts frequencies to match how humans actually hear. We're more sensitive to differences in low frequencies than high frequencies.

**Example:**
- Linear scale: 1000 Hz to 2000 Hz feels like the same jump as 5000 Hz to 6000 Hz
- Mel scale: Adjusts so perceived differences are equal

**Steps to create mel-spectrogram:**

1. **Resample to 16 kHz**
   ```csharp
   // If audio is 44.1 kHz, downsample to 16 kHz
   Vector<T> resampled = ResampleAudio(audioWaveform, originalRate: 44100, targetRate: 16000);
   ```

2. **Apply Short-Time Fourier Transform (STFT)**
   ```csharp
   // Break audio into small overlapping windows (frames)
   // Typical: 25ms windows with 10ms hop
   int windowSize = 400;  // 25ms at 16kHz = 400 samples
   int hopSize = 160;     // 10ms at 16kHz = 160 samples

   // Apply FFT to each window
   Matrix<T> spectrogram = ComputeSTFT(resampled, windowSize, hopSize);
   ```

3. **Convert to power spectrum**
   ```csharp
   // Square the magnitude of complex FFT output
   for (int i = 0; i < spectrogram.Rows; i++)
       for (int j = 0; j < spectrogram.Columns; j++)
           powerSpectrum[i, j] = NumOps.Multiply(spectrogram[i, j], spectrogram[i, j]);
   ```

4. **Apply mel filterbank**
   ```csharp
   // Convert frequency bins to mel scale
   // Whisper typically uses 80 mel bins
   int numMelBins = 80;
   Matrix<T> melFilterbank = CreateMelFilterbank(numMelBins, sampleRate: 16000);
   Matrix<T> melSpectrogram = MatrixHelper.Multiply(powerSpectrum, melFilterbank);
   ```

5. **Apply logarithm**
   ```csharp
   // Convert to decibels (log scale) - matches human perception
   for (int i = 0; i < melSpectrogram.Rows; i++)
       for (int j = 0; j < melSpectrogram.Columns; j++)
       {
           T value = melSpectrogram[i, j];
           // Add small epsilon to avoid log(0)
           T withEpsilon = NumOps.Add(value, NumOps.FromDouble(1e-10));
           melSpectrogram[i, j] = NumOps.Log(withEpsilon);
       }
   ```

**Final shape:** `(time_frames, num_mel_bins)` - typically `(3000, 80)` for a 30-second clip

---

## Whisper Architecture

### Overview

Whisper is an **encoder-decoder transformer** model trained on 680,000 hours of multilingual speech data by OpenAI.

**Key components:**
1. **Audio Encoder**: Processes mel-spectrogram → embeddings
2. **Text Decoder**: Generates text tokens autoregressively

### 1. Audio Encoder

**Architecture:**
```
Input: Mel-Spectrogram (time_frames, 80)
  ↓
Conv1D (kernel=3, stride=1) + GELU
  ↓
Conv1D (kernel=3, stride=2) + GELU  [Downsampling]
  ↓
Positional Embeddings (sinusoidal)
  ↓
Transformer Encoder Blocks (12-32 layers)
  - Multi-Head Self-Attention
  - Layer Normalization
  - Feed-Forward Network (4x expansion)
  - Residual Connections
  ↓
Output: Encoder Hidden States (sequence_length, hidden_dim)
```

**For Beginners:**
- **Conv1D layers**: Extract local patterns in audio (like phonemes)
- **Positional embeddings**: Tell the model where in time each frame occurs
- **Transformer blocks**: Understand relationships between different parts of the audio
- **Output**: Rich representation of the entire audio clip

**Typical dimensions:**
- Whisper Tiny: hidden_dim = 384, 4 encoder layers
- Whisper Base: hidden_dim = 512, 6 encoder layers
- Whisper Small: hidden_dim = 768, 12 encoder layers

### 2. Text Decoder

**Architecture:**
```
Input: Previous Tokens + Encoder Hidden States
  ↓
Token Embedding
  ↓
Positional Embeddings
  ↓
Transformer Decoder Blocks (12-32 layers)
  - Masked Multi-Head Self-Attention (causal)
  - Cross-Attention to Encoder States
  - Layer Normalization
  - Feed-Forward Network
  - Residual Connections
  ↓
Linear Projection to Vocabulary
  ↓
Softmax → Next Token Probabilities
```

**For Beginners:**
- **Causal attention**: Only looks at previous tokens, not future (like reading left-to-right)
- **Cross-attention**: Focuses on relevant parts of the audio for each word
- **Autoregressive**: Generates one token at a time, using previous tokens as context

**Special tokens:**
- `<|startoftranscript|>`: Marks the beginning (token ID: 50258)
- `<|endoftranscript|>`: Marks the end (token ID: 50257)
- `<|en|>`: Language token for English (token ID: 50259)
- `<|notimestamps|>`: Disable timestamp prediction (token ID: 50363)

### 3. Autoregressive Generation Loop

**For Beginners:** The decoder generates text one word at a time, like typing. Each new word depends on all previous words.

**Pseudocode:**
```csharp
// Step 1: Encode audio
Tensor<T> encoderOutput = EncodeAudio(melSpectrogram);

// Step 2: Initialize with start token
List<int> generatedTokens = new List<int> { START_TOKEN };

// Step 3: Generate tokens until end
while (generatedTokens.Count < maxLength)
{
    // Decode with current tokens
    Tensor<T> logits = DecodeTokens(generatedTokens, encoderOutput);

    // Get next token (greedy: pick most probable)
    int nextToken = ArgMax(logits);

    // Add to sequence
    generatedTokens.Add(nextToken);

    // Stop if we hit end token
    if (nextToken == END_TOKEN)
        break;
}

// Step 4: Convert tokens to text
string transcription = TokensToText(generatedTokens);
```

**Key insight:** The encoder runs **once**, but the decoder runs **multiple times** (once per generated token).

---

## Implementation Guide

### Phase 1: Audio Preprocessing Helper

**File:** `src/Audio/AudioPreprocessor.cs`

**Purpose:** Convert raw audio waveform into mel-spectrogram

```csharp
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Audio
{
    /// <summary>
    /// Preprocesses audio waveforms into mel-spectrograms for speech recognition.
    /// </summary>
    /// <typeparam name="T">Numeric type for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <b>For Beginners:</b> This class converts raw audio (a list of numbers representing sound)
    /// into a mel-spectrogram (a 2D image showing frequencies over time). This is like converting
    /// a recording into sheet music that the model can understand.
    /// </remarks>
    public class AudioPreprocessor<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        private readonly int _sampleRate;
        private readonly int _nFft;
        private readonly int _hopLength;
        private readonly int _nMels;

        /// <summary>
        /// Creates a new audio preprocessor with standard Whisper settings.
        /// </summary>
        /// <param name="sampleRate">Audio sample rate in Hz (default: 16000 for Whisper).</param>
        /// <param name="nFft">FFT window size (default: 400 = 25ms at 16kHz).</param>
        /// <param name="hopLength">Number of samples between frames (default: 160 = 10ms at 16kHz).</param>
        /// <param name="nMels">Number of mel frequency bins (default: 80 for Whisper).</param>
        /// <remarks>
        /// <b>Default values explained:</b>
        /// - 16000 Hz: Whisper standard, captures all speech frequencies (0-8kHz)
        /// - 400 samples: 25ms window captures phonemes without smearing
        /// - 160 samples: 10ms hop provides good time resolution
        /// - 80 mels: Balances frequency resolution and computation
        /// </remarks>
        public AudioPreprocessor(
            int sampleRate = 16000,
            int nFft = 400,
            int hopLength = 160,
            int nMels = 80)
        {
            if (sampleRate <= 0)
                throw new ArgumentException("Sample rate must be positive", nameof(sampleRate));
            if (nFft <= 0)
                throw new ArgumentException("FFT size must be positive", nameof(nFft));
            if (hopLength <= 0)
                throw new ArgumentException("Hop length must be positive", nameof(hopLength));
            if (nMels <= 0)
                throw new ArgumentException("Number of mels must be positive", nameof(nMels));

            _sampleRate = sampleRate;
            _nFft = nFft;
            _hopLength = hopLength;
            _nMels = nMels;
        }

        /// <summary>
        /// Converts audio waveform to mel-spectrogram.
        /// </summary>
        /// <param name="waveform">Audio waveform (typically -1.0 to 1.0 range).</param>
        /// <returns>Mel-spectrogram of shape (time_frames, n_mels).</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This method performs the 5-step conversion:
        /// 1. Resample to target rate (if needed)
        /// 2. Apply Short-Time Fourier Transform (STFT)
        /// 3. Compute power spectrum
        /// 4. Apply mel filterbank
        /// 5. Convert to log scale
        /// </remarks>
        public Matrix<T> ComputeMelSpectrogram(Vector<T> waveform)
        {
            // Step 1: Resample is assumed done externally
            // In production, check: if (currentRate != _sampleRate) Resample();

            // Step 2: Compute STFT
            Matrix<Complex<T>> stft = ComputeSTFT(waveform);

            // Step 3: Power spectrum (magnitude squared)
            Matrix<T> powerSpec = ComputePowerSpectrum(stft);

            // Step 4: Apply mel filterbank
            Matrix<T> melFilterbank = CreateMelFilterbank();
            Matrix<T> melSpec = MatrixHelper.Multiply(powerSpec, melFilterbank);

            // Step 5: Log scale (dB)
            return ApplyLogScale(melSpec);
        }

        /// <summary>
        /// Computes Short-Time Fourier Transform using overlapping windows.
        /// </summary>
        private Matrix<Complex<T>> ComputeSTFT(Vector<T> waveform)
        {
            int numFrames = ((waveform.Length - _nFft) / _hopLength) + 1;
            var result = new Matrix<Complex<T>>(numFrames, _nFft / 2 + 1);

            var fft = new FastFourierTransform<T>();
            var window = CreateHannWindow(_nFft);

            for (int frame = 0; frame < numFrames; frame++)
            {
                // Extract frame
                int startIdx = frame * _hopLength;
                var frameData = new Vector<T>(_nFft);

                for (int i = 0; i < _nFft; i++)
                {
                    if (startIdx + i < waveform.Length)
                    {
                        // Apply window function (Hann window reduces spectral leakage)
                        frameData[i] = NumOps.Multiply(waveform[startIdx + i], window[i]);
                    }
                    else
                    {
                        frameData[i] = NumOps.Zero;
                    }
                }

                // Apply FFT
                Vector<Complex<T>> spectrum = fft.Forward(frameData);

                // Store positive frequencies only (FFT is symmetric)
                for (int i = 0; i < _nFft / 2 + 1; i++)
                {
                    result[frame, i] = spectrum[i];
                }
            }

            return result;
        }

        /// <summary>
        /// Creates Hann window for smooth frame edges.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> The Hann window tapers the edges of each frame to zero.
        /// This reduces "spectral leakage" - artifacts from cutting the audio into chunks.
        /// Think of it like fading in and out at frame boundaries.
        /// </remarks>
        private Vector<T> CreateHannWindow(int size)
        {
            var window = new Vector<T>(size);
            for (int i = 0; i < size; i++)
            {
                double value = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (size - 1)));
                window[i] = NumOps.FromDouble(value);
            }
            return window;
        }

        /// <summary>
        /// Computes power spectrum from complex STFT.
        /// </summary>
        private Matrix<T> ComputePowerSpectrum(Matrix<Complex<T>> stft)
        {
            var result = new Matrix<T>(stft.Rows, stft.Columns);

            for (int i = 0; i < stft.Rows; i++)
            {
                for (int j = 0; j < stft.Columns; j++)
                {
                    // Power = Real² + Imaginary²
                    T real = stft[i, j].Real;
                    T imag = stft[i, j].Imaginary;

                    T realSquared = NumOps.Multiply(real, real);
                    T imagSquared = NumOps.Multiply(imag, imag);

                    result[i, j] = NumOps.Add(realSquared, imagSquared);
                }
            }

            return result;
        }

        /// <summary>
        /// Creates mel-scale filterbank matrix.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This creates a set of triangular filters that convert
        /// linear frequency bins to mel-scale. Each mel bin sums several frequency bins
        /// with triangular weights, emphasizing lower frequencies where humans are more sensitive.
        /// </remarks>
        private Matrix<T> CreateMelFilterbank()
        {
            int nFreqs = _nFft / 2 + 1;
            var filterbank = new Matrix<T>(nFreqs, _nMels);

            // Convert Hz to mel scale: mel = 2595 * log10(1 + hz / 700)
            double melMin = HzToMel(0.0);
            double melMax = HzToMel(_sampleRate / 2.0);

            // Create equally-spaced mel points
            double[] melPoints = new double[_nMels + 2];
            for (int i = 0; i < _nMels + 2; i++)
            {
                melPoints[i] = melMin + (melMax - melMin) * i / (_nMels + 1);
            }

            // Convert back to Hz
            double[] hzPoints = melPoints.Select(MelToHz).ToArray();

            // Convert Hz to FFT bin indices
            int[] binPoints = hzPoints.Select(hz =>
                (int)Math.Floor((nFreqs - 1) * hz / (_sampleRate / 2.0))
            ).ToArray();

            // Create triangular filters
            for (int mel = 0; mel < _nMels; mel++)
            {
                int leftBin = binPoints[mel];
                int centerBin = binPoints[mel + 1];
                int rightBin = binPoints[mel + 2];

                // Rising slope
                for (int bin = leftBin; bin < centerBin; bin++)
                {
                    if (bin >= 0 && bin < nFreqs)
                    {
                        double weight = (double)(bin - leftBin) / (centerBin - leftBin);
                        filterbank[bin, mel] = NumOps.FromDouble(weight);
                    }
                }

                // Falling slope
                for (int bin = centerBin; bin < rightBin; bin++)
                {
                    if (bin >= 0 && bin < nFreqs)
                    {
                        double weight = (double)(rightBin - bin) / (rightBin - centerBin);
                        filterbank[bin, mel] = NumOps.FromDouble(weight);
                    }
                }
            }

            return filterbank;
        }

        /// <summary>
        /// Converts frequency in Hz to mel scale.
        /// </summary>
        private double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);

        /// <summary>
        /// Converts mel scale to frequency in Hz.
        /// </summary>
        private double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

        /// <summary>
        /// Applies logarithmic scaling to convert to decibels.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> Converts power to decibels (log scale) to match
        /// human perception of loudness. We hear differences in ratios, not absolute values.
        /// Doubling the power sounds like the same increase whether going from 1→2 or 100→200.
        /// </remarks>
        private Matrix<T> ApplyLogScale(Matrix<T> melSpec)
        {
            var result = new Matrix<T>(melSpec.Rows, melSpec.Columns);
            T epsilon = NumOps.FromDouble(1e-10); // Avoid log(0)

            for (int i = 0; i < melSpec.Rows; i++)
            {
                for (int j = 0; j < melSpec.Columns; j++)
                {
                    T value = NumOps.Add(melSpec[i, j], epsilon);
                    result[i, j] = NumOps.Log(value);
                }
            }

            return result;
        }
    }
}
```

### Phase 2: WhisperModel Wrapper

**File:** `src/Models/Audio/WhisperModel.cs`

**Purpose:** Orchestrate ONNX model for speech-to-text transcription

```csharp
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Audio
{
    /// <summary>
    /// Wrapper for Whisper ONNX model to perform automatic speech recognition.
    /// </summary>
    /// <typeparam name="T">Numeric type for computations.</typeparam>
    /// <remarks>
    /// <b>For Beginners:</b> This class connects all the pieces:
    /// 1. Takes audio waveform
    /// 2. Converts to mel-spectrogram (AudioPreprocessor)
    /// 3. Encodes audio features (ONNX encoder)
    /// 4. Generates text tokens (ONNX decoder, autoregressive)
    /// 5. Converts tokens to readable text
    /// </remarks>
    public class WhisperModel<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        private readonly OnnxModel<T> _onnxModel;
        private readonly AudioPreprocessor<T> _preprocessor;
        private readonly WhisperTokenizer _tokenizer;

        // Special token IDs (from Whisper vocabulary)
        private const int START_TOKEN = 50258;  // <|startoftranscript|>
        private const int END_TOKEN = 50257;    // <|endoftranscript|>
        private const int ENGLISH_TOKEN = 50259; // <|en|>
        private const int NO_TIMESTAMPS = 50363; // <|notimestamps|>

        /// <summary>
        /// Creates a new Whisper model instance.
        /// </summary>
        /// <param name="whisperOnnxPath">Path to the Whisper ONNX model file.</param>
        /// <remarks>
        /// <b>For Beginners:</b> This loads the pre-trained model from disk.
        /// The ONNX file contains all the learned weights from training on 680k hours of audio.
        /// You can download Whisper ONNX models from Hugging Face or convert PyTorch models.
        /// </remarks>
        public WhisperModel(string whisperOnnxPath)
        {
            if (string.IsNullOrWhiteSpace(whisperOnnxPath))
                throw new ArgumentException("Model path cannot be empty", nameof(whisperOnnxPath));
            if (!File.Exists(whisperOnnxPath))
                throw new FileNotFoundException($"Model file not found: {whisperOnnxPath}");

            _onnxModel = new OnnxModel<T>(whisperOnnxPath);
            _preprocessor = new AudioPreprocessor<T>();
            _tokenizer = new WhisperTokenizer();
        }

        /// <summary>
        /// Transcribes audio waveform to text.
        /// </summary>
        /// <param name="audioWaveform">Audio samples at 16kHz, range -1.0 to 1.0.</param>
        /// <param name="maxLength">Maximum number of tokens to generate (default: 448).</param>
        /// <returns>Transcribed text.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is the main method you'll call. Pass in raw audio,
        /// get back transcribed text. The method handles all the complex processing internally.
        ///
        /// <b>Example:</b>
        /// <code>
        /// var audio = LoadWavFile("speech.wav"); // Load audio at 16kHz
        /// var model = new WhisperModel&lt;float&gt;("whisper-tiny.onnx");
        /// string transcription = model.Transcribe(audio);
        /// Console.WriteLine($"Speaker said: {transcription}");
        /// </code>
        /// </remarks>
        public string Transcribe(Tensor<T> audioWaveform, int maxLength = 448)
        {
            if (audioWaveform == null)
                throw new ArgumentNullException(nameof(audioWaveform));
            if (maxLength <= 0)
                throw new ArgumentException("Max length must be positive", nameof(maxLength));

            // Step 1: Preprocess audio to mel-spectrogram
            // Convert from Tensor to Vector for preprocessing
            Vector<T> waveformVector = TensorToVector(audioWaveform);
            Matrix<T> melSpectrogram = _preprocessor.ComputeMelSpectrogram(waveformVector);

            // Step 2: Encode audio features
            Tensor<T> encoderInput = MatrixToTensor(melSpectrogram);
            Tensor<T> encoderOutput = EncodeAudio(encoderInput);

            // Step 3: Autoregressive decoding
            List<int> generatedTokens = GenerateTokens(encoderOutput, maxLength);

            // Step 4: Convert tokens to text
            string transcription = _tokenizer.Decode(generatedTokens);

            return transcription.Trim();
        }

        /// <summary>
        /// Encodes mel-spectrogram using Whisper encoder.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> The encoder runs ONCE per audio clip.
        /// It processes the entire mel-spectrogram and outputs a rich representation
        /// that captures what sounds are present and when they occur.
        /// </remarks>
        private Tensor<T> EncodeAudio(Tensor<T> melSpectrogram)
        {
            // ONNX model expects specific input name (typically "mel" or "input")
            var inputs = new Dictionary<string, Tensor<T>>
            {
                { "mel", melSpectrogram }
            };

            // Run encoder forward pass
            var outputs = _onnxModel.Forward(inputs);

            // Extract encoder hidden states (typically named "encoder_output")
            return outputs["encoder_output"];
        }

        /// <summary>
        /// Generates text tokens autoregressively using Whisper decoder.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the "writing" part of transcription.
        /// The decoder generates one token (word/subword) at a time, using:
        /// - All previously generated tokens (context)
        /// - The encoder output (what sounds are in the audio)
        ///
        /// It's like dictation: the decoder "listens" to the encoded audio and
        /// "types" one word at a time, using previous words for context.
        /// </remarks>
        private List<int> GenerateTokens(Tensor<T> encoderOutput, int maxLength)
        {
            // Initialize with special tokens for English transcription without timestamps
            var tokens = new List<int>
            {
                START_TOKEN,      // <|startoftranscript|>
                ENGLISH_TOKEN,    // <|en|>
                NO_TIMESTAMPS     // <|notimestamps|>
            };

            // Autoregressive generation loop
            for (int step = 0; step < maxLength; step++)
            {
                // Prepare decoder inputs
                Tensor<T> tokenIds = TokensToTensor(tokens);

                var inputs = new Dictionary<string, Tensor<T>>
                {
                    { "tokens", tokenIds },              // Previous tokens
                    { "encoder_output", encoderOutput }  // Audio features
                };

                // Run decoder forward pass
                var outputs = _onnxModel.Forward(inputs);
                Tensor<T> logits = outputs["logits"];

                // Get next token (greedy decoding: pick most probable)
                int nextToken = GetMostProbableToken(logits);

                // Stop if we hit end token
                if (nextToken == END_TOKEN)
                    break;

                // Add to sequence
                tokens.Add(nextToken);
            }

            return tokens;
        }

        /// <summary>
        /// Selects the most probable next token (greedy decoding).
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> The decoder outputs probabilities for every possible token.
        /// Greedy decoding simply picks the token with highest probability.
        ///
        /// <b>Alternative approaches:</b>
        /// - Beam search: Keep top-K candidates, explore multiple paths
        /// - Sampling: Randomly select based on probabilities (more creative but less accurate)
        /// </remarks>
        private int GetMostProbableToken(Tensor<T> logits)
        {
            // Logits shape: (batch=1, sequence_length, vocab_size)
            // We want the last position (most recent token prediction)

            int vocabSize = logits.Shape[logits.Shape.Length - 1];
            int lastPos = logits.Shape[logits.Shape.Length - 2] - 1;

            int maxIdx = 0;
            T maxValue = NumOps.FromDouble(double.NegativeInfinity);

            for (int i = 0; i < vocabSize; i++)
            {
                T value = logits[0, lastPos, i]; // Get logit for token i
                if (NumOps.GreaterThan(value, maxValue))
                {
                    maxValue = value;
                    maxIdx = i;
                }
            }

            return maxIdx;
        }

        /// <summary>
        /// Converts token IDs to tensor for ONNX input.
        /// </summary>
        private Tensor<T> TokensToTensor(List<int> tokens)
        {
            int[] shape = new int[] { 1, tokens.Count }; // (batch_size=1, sequence_length)
            var data = new Vector<T>(tokens.Count);

            for (int i = 0; i < tokens.Count; i++)
            {
                data[i] = NumOps.FromDouble(tokens[i]);
            }

            return new Tensor<T>(shape, data);
        }

        /// <summary>
        /// Helper to convert 1D Tensor to Vector.
        /// </summary>
        private Vector<T> TensorToVector(Tensor<T> tensor)
        {
            if (tensor.Shape.Length != 1)
                throw new ArgumentException("Expected 1D tensor for audio waveform");

            var result = new Vector<T>(tensor.Shape[0]);
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = tensor[i];
            }
            return result;
        }

        /// <summary>
        /// Helper to convert Matrix to Tensor.
        /// </summary>
        private Tensor<T> MatrixToTensor(Matrix<T> matrix)
        {
            int[] shape = new int[] { 1, matrix.Rows, matrix.Columns }; // (batch, time, mels)
            var data = new Vector<T>(matrix.Rows * matrix.Columns);

            int idx = 0;
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    data[idx++] = matrix[i, j];
                }
            }

            return new Tensor<T>(shape, data);
        }
    }

    /// <summary>
    /// Handles tokenization and detokenization for Whisper.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This converts between:
    /// - Token IDs (numbers the model uses): [50258, 50259, 1234, 5678, ...]
    /// - Text (what humans read): "Hello world"
    ///
    /// Whisper uses BPE (Byte Pair Encoding) tokenization with ~51k vocabulary.
    /// </remarks>
    internal class WhisperTokenizer
    {
        // In production, load from tokenizer.json or similar
        // For now, simplified implementation

        public string Decode(List<int> tokens)
        {
            // Filter out special tokens
            var textTokens = tokens.Where(t => t < 50257).ToList();

            // TODO: Implement actual BPE decoding
            // For now, placeholder that would be replaced with real tokenizer
            return "[Transcribed text would appear here]";
        }
    }
}
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/UnitTests/Models/WhisperModelTests.cs`

**Purpose:** Test logic with mocked ONNX runtime

```csharp
using Xunit;
using Moq;
using AiDotNet.Models.Audio;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Models
{
    public class WhisperModelTests
    {
        [Fact]
        public void Transcribe_CallsEncoderOnce()
        {
            // Arrange
            var mockOnnx = new Mock<OnnxModel<float>>();
            var audio = CreateDummyAudio(16000); // 1 second at 16kHz

            // Mock encoder output
            var encoderOutput = new Tensor<float>(new int[] { 1, 1500, 384 });
            mockOnnx.Setup(m => m.Forward(It.Is<Dictionary<string, Tensor<float>>>(
                d => d.ContainsKey("mel")
            ))).Returns(new Dictionary<string, Tensor<float>>
            {
                { "encoder_output", encoderOutput }
            });

            // Mock decoder output (end token immediately)
            var decoderOutput = CreateEndTokenLogits();
            mockOnnx.Setup(m => m.Forward(It.Is<Dictionary<string, Tensor<float>>>(
                d => d.ContainsKey("tokens")
            ))).Returns(new Dictionary<string, Tensor<float>>
            {
                { "logits", decoderOutput }
            });

            var model = new WhisperModel<float>("dummy.onnx", mockOnnx.Object);

            // Act
            model.Transcribe(audio);

            // Assert
            mockOnnx.Verify(m => m.Forward(It.Is<Dictionary<string, Tensor<float>>>(
                d => d.ContainsKey("mel")
            )), Times.Once, "Encoder should be called exactly once");
        }

        [Fact]
        public void Transcribe_CallsDecoderAutoregressively()
        {
            // Arrange
            var mockOnnx = new Mock<OnnxModel<float>>();
            var audio = CreateDummyAudio(16000);

            int decoderCallCount = 0;

            mockOnnx.Setup(m => m.Forward(It.Is<Dictionary<string, Tensor<float>>>(
                d => d.ContainsKey("tokens")
            ))).Returns(() =>
            {
                decoderCallCount++;
                // Return end token on 5th call
                return decoderCallCount >= 5
                    ? CreateEndTokenLogits()
                    : CreateContinueTokenLogits(decoderCallCount);
            });

            var model = new WhisperModel<float>("dummy.onnx", mockOnnx.Object);

            // Act
            model.Transcribe(audio);

            // Assert
            Assert.Equal(5, decoderCallCount);
        }

        private Tensor<float> CreateDummyAudio(int samples)
        {
            var data = new Vector<float>(samples);
            for (int i = 0; i < samples; i++)
            {
                // Simple sine wave
                data[i] = (float)Math.Sin(2.0 * Math.PI * 440.0 * i / 16000.0);
            }
            return new Tensor<float>(new int[] { samples }, data);
        }

        private Dictionary<string, Tensor<float>> CreateEndTokenLogits()
        {
            var logits = new Tensor<float>(new int[] { 1, 1, 51865 }); // vocab size
            logits[0, 0, 50257] = 100.0f; // High confidence for END_TOKEN
            return new Dictionary<string, Tensor<float>> { { "logits", logits } };
        }

        private Dictionary<string, Tensor<float>> CreateContinueTokenLogits(int tokenId)
        {
            var logits = new Tensor<float>(new int[] { 1, 1, 51865 });
            logits[0, 0, tokenId] = 100.0f; // Return different token each time
            return new Dictionary<string, Tensor<float>> { { "logits", logits } };
        }
    }
}
```

### Integration Tests

**File:** `tests/IntegrationTests/Models/WhisperModelIntegrationTests.cs`

**Purpose:** Test with real ONNX model and audio files

```csharp
using Xunit;
using AiDotNet.Models.Audio;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.Models
{
    public class WhisperModelIntegrationTests
    {
        // Test is ignored if model file doesn't exist
        [Fact(Skip = "Requires Whisper ONNX model - download from Hugging Face")]
        public void Transcribe_RealModel_ProducesAccurateText()
        {
            // Arrange
            string modelPath = "whisper-tiny.en.onnx";
            string audioPath = "test_audio_hello_world.wav";

            if (!File.Exists(modelPath))
            {
                Assert.True(false, $"Model not found. Download from: https://huggingface.co/onnx-models/whisper-tiny.en");
            }

            if (!File.Exists(audioPath))
            {
                Assert.True(false, $"Test audio not found. Create a WAV file of someone saying 'Hello world'");
            }

            var model = new WhisperModel<float>(modelPath);
            var audio = LoadWavFile(audioPath);

            // Act
            string transcription = model.Transcribe(audio);

            // Assert
            string normalizedTranscription = transcription.ToLowerInvariant()
                .Replace(".", "")
                .Replace(",", "")
                .Replace("!", "")
                .Trim();

            Assert.Contains("hello world", normalizedTranscription);
        }

        [Fact(Skip = "Requires model file")]
        public void Transcribe_LongerAudio_HandlesMultipleSentences()
        {
            // Arrange
            string modelPath = "whisper-tiny.en.onnx";
            string audioPath = "test_audio_paragraph.wav";

            var model = new WhisperModel<float>(modelPath);
            var audio = LoadWavFile(audioPath);

            // Act
            string transcription = model.Transcribe(audio, maxLength: 896); // Longer max length

            // Assert
            Assert.NotEmpty(transcription);
            Assert.True(transcription.Length > 50, "Should transcribe multiple sentences");
        }

        private Tensor<float> LoadWavFile(string path)
        {
            // Use NAudio or similar library to load WAV
            // For now, simplified placeholder

            using (var reader = new System.IO.BinaryReader(File.OpenRead(path)))
            {
                // Skip WAV header (44 bytes)
                reader.ReadBytes(44);

                // Read 16-bit PCM samples
                var samples = new List<float>();
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    short sample = reader.ReadInt16();
                    samples.Add(sample / 32768.0f); // Normalize to -1.0 to 1.0
                }

                var data = new Vector<float>(samples.Count);
                for (int i = 0; i < samples.Count; i++)
                {
                    data[i] = samples[i];
                }

                return new Tensor<float>(new int[] { samples.Count }, data);
            }
        }
    }
}
```

---

## Common Pitfalls

### 1. Wrong Sampling Rate

**Problem:** Audio is 44.1kHz but Whisper expects 16kHz
**Solution:** Always resample before processing
```csharp
// Use a resampling library or implement simple linear interpolation
Vector<T> resampled = ResampleAudio(audio, 44100, 16000);
```

### 2. Incorrect Mel-Spectrogram Shape

**Problem:** ONNX model expects specific shape `(batch, time, 80)`
**Solution:** Check model input requirements
```csharp
// Whisper expects: (1, num_frames, 80)
// Ensure AudioPreprocessor outputs (num_frames, 80)
// Then add batch dimension: (1, num_frames, 80)
```

### 3. Forgetting to Apply Window Function

**Problem:** Harsh frame boundaries cause spectral leakage
**Solution:** Always apply Hann window in STFT
```csharp
frameData[i] = NumOps.Multiply(waveform[startIdx + i], window[i]);
```

### 4. Log of Zero

**Problem:** `log(0) = -∞` causes NaN errors
**Solution:** Add small epsilon before log
```csharp
T epsilon = NumOps.FromDouble(1e-10);
T value = NumOps.Add(melSpec[i, j], epsilon);
result[i, j] = NumOps.Log(value);
```

### 5. Infinite Decoder Loop

**Problem:** Model never generates END_TOKEN
**Solution:** Always have max length failsafe
```csharp
for (int step = 0; step < maxLength; step++)
{
    // Generate token
    if (nextToken == END_TOKEN) break;
}
```

### 6. Wrong Token Embedding Dimension

**Problem:** Decoder expects specific input shape
**Solution:** Check ONNX model signature
```csharp
// Use tool like Netron to visualize ONNX model inputs/outputs
// Ensure token tensor matches expected shape
```

---

## Resources

### Papers
- **Whisper (OpenAI, 2022)**: "Robust Speech Recognition via Large-Scale Weak Supervision"
- **Conformer**: "Conformer: Convolution-augmented Transformer for Speech Recognition"

### Code Examples
- OpenAI Whisper (PyTorch): https://github.com/openai/whisper
- Whisper.cpp (C++): https://github.com/ggerganov/whisper.cpp
- ONNX Runtime examples: https://github.com/microsoft/onnxruntime

### ONNX Models
- Hugging Face ONNX models: https://huggingface.co/models?library=onnx
- Convert PyTorch to ONNX: `torch.onnx.export()`

### Audio Processing
- Librosa documentation (Python, but concepts apply): https://librosa.org/
- Understanding STFT: https://ccrma.stanford.edu/~jos/sasp/Short_Time_Fourier_Transform.html
- Mel scale explained: https://en.wikipedia.org/wiki/Mel_scale

---

## Next Steps

After implementing Issue #269, you'll have:
1. Complete audio preprocessing pipeline
2. Working ASR inference with Whisper
3. Foundation for other audio models (TTS, audio generation)

The same concepts apply to:
- **Issue #270 (TTS)**: Reverse process - text → spectrogram → audio
- **Issue #271 (Audio Gen)**: Language model → audio codes → audio

Good luck with your implementation!
