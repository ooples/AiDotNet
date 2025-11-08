# Issue #397: Junior Developer Implementation Guide
## Audio AI Models (Wav2Vec2, Whisper, MusicGen)

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Audio Processing Fundamentals](#audio-processing-fundamentals)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Strategy](#implementation-strategy)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Understanding the Problem

### What Are We Building?

We're implementing support for **audio AI models** that can:
- **Speech Recognition**: Convert audio to text (Whisper, Wav2Vec2)
- **Audio Classification**: Identify sounds, speakers, or music genres
- **Music Generation**: Create new audio from text descriptions (MusicGen)
- **Audio Feature Extraction**: Extract meaningful representations from audio

### Why Audio Models Are Special

Unlike images (2D) or text (1D sequences), audio has unique characteristics:
1. **Temporal dimension**: Audio unfolds over time
2. **Frequency dimension**: Different pitches/frequencies matter
3. **High sampling rates**: CD quality is 44,100 samples per second
4. **Spectral representations**: Often processed as spectrograms (time-frequency images)

### Real-World Use Cases

- **Voice assistants**: "Hey Siri, what's the weather?"
- **Transcription services**: Converting meetings to text
- **Music analysis**: Genre classification, mood detection
- **Audio generation**: Text-to-speech, music composition
- **Sound event detection**: Glass breaking, dog barking, etc.

---

## Audio Processing Fundamentals

### Understanding Audio Data

#### 1. Waveform (Time Domain)
```csharp
/// <summary>
/// Raw audio as amplitude values over time.
/// Shape: [samples] or [channels, samples]
/// </summary>
/// <remarks>
/// For Beginners:
/// Audio waveform is like a graph showing how loud the sound is at each moment.
/// - Amplitude: How loud (y-axis)
/// - Time: When it happens (x-axis)
/// - Sample rate: How many measurements per second (e.g., 16000 Hz)
///
/// Example:
/// 1 second of audio at 16kHz = 16,000 samples
/// Stereo audio = 2 channels (left and right)
/// </remarks>
public class AudioWaveform<T>
{
    // Tensor shape: [channels, samples]
    // For mono: [1, 16000] for 1 second at 16kHz
    // For stereo: [2, 16000] for 1 second at 16kHz
    public Tensor<T> Data { get; set; }

    public int SampleRate { get; set; }  // Samples per second (Hz)
    public int Channels { get; set; }     // 1 = mono, 2 = stereo
    public int Samples { get; set; }      // Total samples

    public double DurationSeconds => Samples / (double)SampleRate;
}
```

#### 2. Spectrogram (Frequency Domain)
```csharp
/// <summary>
/// Audio represented as time-frequency image.
/// Shows which frequencies are present at each moment in time.
/// Shape: [frequency_bins, time_frames]
/// </summary>
/// <remarks>
/// For Beginners:
/// A spectrogram is like a heatmap showing:
/// - X-axis: Time (when)
/// - Y-axis: Frequency (pitch - low to high)
/// - Color/intensity: How strong that frequency is
///
/// Think of it like sheet music:
/// - Time flows left to right
/// - Pitch (frequency) is up/down
/// - Loudness is brightness
///
/// Why spectrograms?
/// - Neural networks can treat them like images
/// - Easier to see patterns (speech, music notes, etc.)
/// - Many audio models (Wav2Vec2, Whisper) use them internally
/// </remarks>
public class Spectrogram<T>
{
    // Shape: [frequency_bins, time_frames]
    // frequency_bins: Number of frequency bands (e.g., 80 mel bands)
    // time_frames: Number of time windows
    public Tensor<T> Data { get; set; }

    public int FrequencyBins { get; set; }  // Y-axis: frequencies
    public int TimeFrames { get; set; }     // X-axis: time windows
    public int HopLength { get; set; }      // Samples between frames
    public int WindowSize { get; set; }     // FFT window size

    public double TimeResolution => HopLength / (double)SampleRate;
}
```

#### 3. Mel Spectrogram
```csharp
/// <summary>
/// Spectrogram with frequency bins spaced according to human hearing (mel scale).
/// Shape: [mel_bins, time_frames]
/// </summary>
/// <remarks>
/// For Beginners:
/// Humans don't hear frequencies linearly - we're more sensitive to low frequencies.
/// The mel scale mimics human hearing:
/// - More detail in low frequencies (speech, bass)
/// - Less detail in high frequencies (we can't distinguish as well)
///
/// Example:
/// - 100 Hz to 200 Hz: Big perceptual difference (octave)
/// - 10,000 Hz to 10,100 Hz: Barely noticeable
///
/// Mel spectrograms are preferred for speech/music because they match how we hear.
/// </remarks>
public class MelSpectrogram<T>
{
    public Tensor<T> Data { get; set; }  // Shape: [mel_bins, time_frames]

    public int MelBins { get; set; }     // Typically 80 or 128
    public int SampleRate { get; set; }
    public int HopLength { get; set; }
    public int WindowSize { get; set; }

    // Mel scale conversion
    public double FrequencyMin { get; set; }  // Typically 0 Hz
    public double FrequencyMax { get; set; }  // Typically 8000 Hz (half sample rate)
}
```

### Audio Preprocessing Pipeline

```csharp
/// <summary>
/// Standard pipeline for audio preprocessing.
/// </summary>
/// <remarks>
/// For Beginners:
/// Converting raw audio to model input happens in stages:
///
/// 1. Load audio file → waveform (raw samples)
/// 2. Resample → match model's expected sample rate
/// 3. Convert to mono → single channel if needed
/// 4. Normalize → scale to [-1, 1] or [0, 1]
/// 5. Extract features → mel spectrogram
/// 6. Normalize features → zero mean, unit variance
///
/// Why each step?
/// - Resampling: Models trained on specific rates (16kHz for Whisper)
/// - Mono conversion: Most models use single channel
/// - Normalization: Prevents numerical instability
/// - Feature extraction: Models work better with spectrograms
/// </remarks>
public class AudioPreprocessor<T>
{
    private readonly int _targetSampleRate;
    private readonly int _melBins;
    private readonly int _hopLength;
    private readonly int _windowSize;

    public AudioPreprocessor(
        int targetSampleRate = 16000,
        int melBins = 80,
        int hopLength = 160,
        int windowSize = 400)
    {
        Guard.Positive(targetSampleRate, nameof(targetSampleRate));
        Guard.Positive(melBins, nameof(melBins));
        Guard.Positive(hopLength, nameof(hopLength));
        Guard.Positive(windowSize, nameof(windowSize));

        _targetSampleRate = targetSampleRate;
        _melBins = melBins;
        _hopLength = hopLength;
        _windowSize = windowSize;
    }

    /// <summary>
    /// Process raw audio to mel spectrogram.
    /// </summary>
    public MelSpectrogram<T> Process(AudioWaveform<T> audio)
    {
        Guard.NotNull(audio, nameof(audio));

        // Step 1: Resample if needed
        var resampled = Resample(audio, _targetSampleRate);

        // Step 2: Convert to mono if stereo
        var mono = ConvertToMono(resampled);

        // Step 3: Normalize amplitude to [-1, 1]
        var normalized = NormalizeAmplitude(mono);

        // Step 4: Compute mel spectrogram
        var melSpec = ComputeMelSpectrogram(
            normalized,
            _melBins,
            _hopLength,
            _windowSize);

        // Step 5: Apply log scaling (human hearing is logarithmic)
        var logMelSpec = ApplyLogScaling(melSpec);

        // Step 6: Normalize features
        var normalizedSpec = NormalizeFeatures(logMelSpec);

        return normalizedSpec;
    }

    private AudioWaveform<T> Resample(AudioWaveform<T> audio, int targetRate)
    {
        if (audio.SampleRate == targetRate)
            return audio;

        // Resample using linear interpolation
        // For production: use sinc interpolation or polyphase filtering
        double ratio = (double)targetRate / audio.SampleRate;
        int newSamples = (int)(audio.Samples * ratio);

        var resampled = new Tensor<T>(new[] { audio.Channels, newSamples });

        // Simple linear interpolation (production should use better methods)
        for (int ch = 0; ch < audio.Channels; ch++)
        {
            for (int i = 0; i < newSamples; i++)
            {
                double sourceIndex = i / ratio;
                int idx0 = (int)sourceIndex;
                int idx1 = Math.Min(idx0 + 1, audio.Samples - 1);
                double frac = sourceIndex - idx0;

                dynamic val0 = audio.Data[ch, idx0];
                dynamic val1 = audio.Data[ch, idx1];
                resampled[ch, i] = (T)(object)(val0 * (1 - frac) + val1 * frac);
            }
        }

        return new AudioWaveform<T>
        {
            Data = resampled,
            SampleRate = targetRate,
            Channels = audio.Channels,
            Samples = newSamples
        };
    }

    private AudioWaveform<T> ConvertToMono(AudioWaveform<T> audio)
    {
        if (audio.Channels == 1)
            return audio;

        // Average all channels
        var mono = new Tensor<T>(new[] { 1, audio.Samples });

        for (int i = 0; i < audio.Samples; i++)
        {
            dynamic sum = (T)(object)0.0;
            for (int ch = 0; ch < audio.Channels; ch++)
            {
                sum += audio.Data[ch, i];
            }
            mono[0, i] = (T)(object)(sum / audio.Channels);
        }

        return new AudioWaveform<T>
        {
            Data = mono,
            SampleRate = audio.SampleRate,
            Channels = 1,
            Samples = audio.Samples
        };
    }

    private AudioWaveform<T> NormalizeAmplitude(AudioWaveform<T> audio)
    {
        // Scale to [-1, 1] based on max absolute value
        dynamic maxAbs = (T)(object)0.0;

        for (int ch = 0; ch < audio.Channels; ch++)
        {
            for (int i = 0; i < audio.Samples; i++)
            {
                dynamic val = audio.Data[ch, i];
                dynamic abs = Math.Abs(val);
                if (abs > maxAbs)
                    maxAbs = abs;
            }
        }

        if (maxAbs == (T)(object)0.0)
            return audio;  // Silence

        var normalized = audio.Data.Clone();
        for (int ch = 0; ch < audio.Channels; ch++)
        {
            for (int i = 0; i < audio.Samples; i++)
            {
                normalized[ch, i] = (T)(object)(audio.Data[ch, i] / maxAbs);
            }
        }

        return new AudioWaveform<T>
        {
            Data = normalized,
            SampleRate = audio.SampleRate,
            Channels = audio.Channels,
            Samples = audio.Samples
        };
    }

    private MelSpectrogram<T> ComputeMelSpectrogram(
        AudioWaveform<T> audio,
        int melBins,
        int hopLength,
        int windowSize)
    {
        // This is simplified - production should use FFT library
        int numFrames = 1 + (audio.Samples - windowSize) / hopLength;
        var spectrogram = new Tensor<T>(new[] { melBins, numFrames });

        // For each time frame
        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * hopLength;

            // Extract window
            var window = new T[windowSize];
            for (int i = 0; i < windowSize; i++)
            {
                if (start + i < audio.Samples)
                {
                    window[i] = audio.Data[0, start + i];
                }
            }

            // Apply Hann window to reduce spectral leakage
            ApplyHannWindow(window);

            // Compute FFT (simplified - use Math.NET or similar in production)
            var fft = ComputeFFT(window);

            // Convert to mel scale
            var melEnergies = ConvertToMelScale(fft, melBins, audio.SampleRate);

            // Store in spectrogram
            for (int mel = 0; mel < melBins; mel++)
            {
                spectrogram[mel, frame] = melEnergies[mel];
            }
        }

        return new MelSpectrogram<T>
        {
            Data = spectrogram,
            MelBins = melBins,
            SampleRate = audio.SampleRate,
            HopLength = hopLength,
            WindowSize = windowSize
        };
    }

    private void ApplyHannWindow(T[] samples)
    {
        int n = samples.Length;
        for (int i = 0; i < n; i++)
        {
            double window = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (n - 1)));
            samples[i] = (T)(object)(Convert.ToDouble(samples[i]) * window);
        }
    }

    private T[] ComputeFFT(T[] samples)
    {
        // Simplified placeholder - use proper FFT library in production
        // (Math.NET Numerics, FFTW wrapper, etc.)
        int n = samples.Length;
        var magnitudes = new T[n / 2];

        // This is a placeholder - real FFT required
        for (int k = 0; k < n / 2; k++)
        {
            magnitudes[k] = (T)(object)0.0;
        }

        return magnitudes;
    }

    private T[] ConvertToMelScale(T[] fftMagnitudes, int melBins, int sampleRate)
    {
        var melEnergies = new T[melBins];

        // Mel scale conversion: mel = 2595 * log10(1 + f/700)
        double fMin = 0;
        double fMax = sampleRate / 2.0;
        double melMin = HzToMel(fMin);
        double melMax = HzToMel(fMax);

        // Create mel filterbank
        for (int mel = 0; mel < melBins; mel++)
        {
            double energy = 0;

            // Map mel bin to frequency range
            double melCenter = melMin + (melMax - melMin) * mel / (melBins - 1);
            double fCenter = MelToHz(melCenter);

            // Sum FFT bins that fall in this mel bin (simplified)
            // Real implementation uses triangular filters
            int fftBinStart = (int)(fCenter * fftMagnitudes.Length / (sampleRate / 2.0));
            int fftBinEnd = Math.Min(fftBinStart + 10, fftMagnitudes.Length);

            for (int i = fftBinStart; i < fftBinEnd; i++)
            {
                energy += Convert.ToDouble(fftMagnitudes[i]);
            }

            melEnergies[mel] = (T)(object)energy;
        }

        return melEnergies;
    }

    private double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
    private double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

    private MelSpectrogram<T> ApplyLogScaling(MelSpectrogram<T> spec)
    {
        var logged = spec.Data.Clone();

        for (int mel = 0; mel < spec.MelBins; mel++)
        {
            for (int frame = 0; frame < spec.TimeFrames; frame++)
            {
                dynamic val = spec.Data[mel, frame];
                // Log scaling with small epsilon to avoid log(0)
                logged[mel, frame] = (T)(object)Math.Log(val + 1e-10);
            }
        }

        return new MelSpectrogram<T>
        {
            Data = logged,
            MelBins = spec.MelBins,
            SampleRate = spec.SampleRate,
            HopLength = spec.HopLength,
            WindowSize = spec.WindowSize
        };
    }

    private MelSpectrogram<T> NormalizeFeatures(MelSpectrogram<T> spec)
    {
        // Normalize to zero mean, unit variance (per frequency bin)
        var normalized = spec.Data.Clone();

        for (int mel = 0; mel < spec.MelBins; mel++)
        {
            // Compute mean and std for this frequency bin
            double mean = 0;
            double variance = 0;

            for (int frame = 0; frame < spec.TimeFrames; frame++)
            {
                mean += Convert.ToDouble(spec.Data[mel, frame]);
            }
            mean /= spec.TimeFrames;

            for (int frame = 0; frame < spec.TimeFrames; frame++)
            {
                double diff = Convert.ToDouble(spec.Data[mel, frame]) - mean;
                variance += diff * diff;
            }
            variance /= spec.TimeFrames;
            double std = Math.Sqrt(variance);

            if (std < 1e-10)
                std = 1.0;  // Avoid division by zero

            // Normalize
            for (int frame = 0; frame < spec.TimeFrames; frame++)
            {
                double val = Convert.ToDouble(spec.Data[mel, frame]);
                normalized[mel, frame] = (T)(object)((val - mean) / std);
            }
        }

        return new MelSpectrogram<T>
        {
            Data = normalized,
            MelBins = spec.MelBins,
            SampleRate = spec.SampleRate,
            HopLength = spec.HopLength,
            WindowSize = spec.WindowSize
        };
    }
}
```

---

## Architecture Overview

### Model Taxonomy

```
Audio AI Models
├── Speech Recognition (Audio → Text)
│   ├── Wav2Vec2 (Self-supervised learning)
│   ├── Whisper (Encoder-decoder transformer)
│   └── Conformer (Hybrid CNN + Transformer)
│
├── Audio Classification (Audio → Labels)
│   ├── Audio Spectrogram Transformer (AST)
│   ├── ResNet-based (treat spectrogram as image)
│   └── PANNS (Pretrained Audio Neural Networks)
│
└── Audio Generation (Text/Conditions → Audio)
    ├── MusicGen (Text-to-music)
    ├── AudioLM (Continuation, inpainting)
    └── Jukebox (High-fidelity music)
```

### Wav2Vec2 Architecture

```csharp
/// <summary>
/// Wav2Vec2 model for speech recognition.
/// Uses self-supervised learning with masked prediction.
/// </summary>
/// <remarks>
/// For Beginners:
/// Wav2Vec2 learns audio representations by:
/// 1. Encoding raw audio waveform with CNN
/// 2. Masking parts of the encoding
/// 3. Predicting masked parts (like BERT for text)
/// 4. Fine-tuning on labeled speech data
///
/// Architecture:
/// Input: Raw waveform [samples]
/// → CNN Feature Encoder: [samples] → [features, time_steps]
/// → Transformer Encoder: [features, time_steps] → [hidden_dim, time_steps]
/// → CTC Head: [hidden_dim, time_steps] → [vocab_size, time_steps]
/// Output: Character/word probabilities for each time step
///
/// Key innovation: Works directly on raw audio, no spectrogram needed
/// </remarks>
public class Wav2Vec2Model<T> : IAudioModel<T>
{
    private readonly Wav2Vec2Config _config;
    private readonly CNNFeatureEncoder<T> _featureEncoder;
    private readonly TransformerEncoder<T> _transformer;
    private readonly CTCHead<T> _ctcHead;

    public Wav2Vec2Model(Wav2Vec2Config config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        // Feature encoder: raw waveform → latent features
        _featureEncoder = new CNNFeatureEncoder<T>(
            inputChannels: 1,
            outputChannels: config.HiddenSize,
            kernelSizes: new[] { 10, 3, 3, 3, 3, 2, 2 },
            strides: new[] { 5, 2, 2, 2, 2, 2, 2 },
            activations: "gelu");

        // Transformer encoder: process temporal context
        _transformer = new TransformerEncoder<T>(
            numLayers: config.NumLayers,
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            intermediateSize: config.IntermediateSize,
            dropoutRate: config.DropoutRate);

        // CTC head: predict characters/tokens
        _ctcHead = new CTCHead<T>(
            inputSize: config.HiddenSize,
            vocabSize: config.VocabSize);
    }

    public AudioModelOutput<T> Forward(Tensor<T> waveform)
    {
        Guard.NotNull(waveform, nameof(waveform));

        // waveform shape: [batch, samples]

        // Step 1: Extract features with CNN
        // Output: [batch, hidden_size, time_steps]
        var features = _featureEncoder.Forward(waveform);

        // Step 2: Transpose for transformer [batch, time_steps, hidden_size]
        var transposed = features.Transpose(1, 2);

        // Step 3: Apply transformer
        var encoded = _transformer.Forward(transposed);

        // Step 4: CTC prediction
        // Output: [batch, time_steps, vocab_size]
        var logits = _ctcHead.Forward(encoded);

        return new AudioModelOutput<T>
        {
            Logits = logits,
            HiddenStates = encoded,
            Features = features
        };
    }
}

public class Wav2Vec2Config
{
    public int HiddenSize { get; set; } = 768;
    public int NumLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 12;
    public int IntermediateSize { get; set; } = 3072;
    public int VocabSize { get; set; } = 32;  // Characters/tokens
    public double DropoutRate { get; set; } = 0.1;
}
```

### Whisper Architecture

```csharp
/// <summary>
/// OpenAI Whisper model for robust speech recognition.
/// Encoder-decoder transformer trained on 680k hours of multilingual data.
/// </summary>
/// <remarks>
/// For Beginners:
/// Whisper is like a translator:
/// - Encoder: Converts audio spectrogram → meaning representation
/// - Decoder: Converts meaning → text tokens
///
/// Architecture:
/// Input: Mel spectrogram [mel_bins=80, time_frames]
/// → Encoder: Conv layers + Transformer → [hidden_dim, time_frames/2]
/// → Decoder: Transformer with cross-attention → [vocab_size, seq_len]
/// Output: Text tokens (transcription)
///
/// Special features:
/// - Multilingual (99 languages)
/// - Multitask (transcribe, translate, timestamp, language detection)
/// - Robust to accents, background noise
/// - Zero-shot (no fine-tuning needed)
/// </remarks>
public class WhisperModel<T> : IAudioModel<T>
{
    private readonly WhisperConfig _config;
    private readonly WhisperEncoder<T> _encoder;
    private readonly WhisperDecoder<T> _decoder;

    public WhisperModel(WhisperConfig config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        _encoder = new WhisperEncoder<T>(
            numLayers: config.EncoderLayers,
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            melBins: config.MelBins);

        _decoder = new WhisperDecoder<T>(
            numLayers: config.DecoderLayers,
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            vocabSize: config.VocabSize);
    }

    public AudioModelOutput<T> Forward(
        MelSpectrogram<T> melSpec,
        Tensor<T>? decoderInput = null)
    {
        Guard.NotNull(melSpec, nameof(melSpec));

        // Step 1: Encode mel spectrogram
        // Input: [batch, mel_bins=80, time_frames=3000]
        // Output: [batch, time_frames/2, hidden_size]
        var encoderOutput = _encoder.Forward(melSpec.Data);

        // Step 2: Decode to text
        if (decoderInput == null)
        {
            // Generate from start token
            decoderInput = CreateStartTokens(encoderOutput.Shape[0]);
        }

        // Output: [batch, seq_len, vocab_size]
        var decoderOutput = _decoder.Forward(
            decoderInput,
            encoderHiddenStates: encoderOutput);

        return new AudioModelOutput<T>
        {
            Logits = decoderOutput,
            EncoderHiddenStates = encoderOutput,
            DecoderHiddenStates = decoderOutput
        };
    }

    private Tensor<T> CreateStartTokens(int batchSize)
    {
        // Create [batch, 1] tensor with start-of-transcript token
        var tokens = new Tensor<T>(new[] { batchSize, 1 });
        for (int i = 0; i < batchSize; i++)
        {
            tokens[i, 0] = (T)(object)_config.StartOfTranscriptToken;
        }
        return tokens;
    }
}

public class WhisperConfig
{
    public int EncoderLayers { get; set; } = 6;
    public int DecoderLayers { get; set; } = 6;
    public int HiddenSize { get; set; } = 384;
    public int NumAttentionHeads { get; set; } = 6;
    public int MelBins { get; set; } = 80;
    public int VocabSize { get; set; } = 51865;
    public int StartOfTranscriptToken { get; set; } = 50258;
    public int MaxSourceLength { get; set; } = 3000;  // Time frames
    public int MaxTargetLength { get; set; } = 448;   // Text tokens
}
```

### MusicGen Architecture

```csharp
/// <summary>
/// Meta's MusicGen model for text-to-music generation.
/// Uses LM + audio codec for high-quality music synthesis.
/// </summary>
/// <remarks>
/// For Beginners:
/// MusicGen generates music in two stages:
///
/// 1. Language Model Stage:
///    - Input: Text description ("upbeat jazz piano")
///    - Output: Sequence of audio codes (compressed representation)
///    - Uses transformer to predict codes autoregressively
///
/// 2. Decoding Stage:
///    - Input: Audio codes
///    - Output: Waveform (actual audio)
///    - Uses EnCodec to decompress codes to audio
///
/// Key innovation: Multi-codebook modeling
/// - Audio encoded as multiple parallel codebooks
/// - Allows high-quality 32kHz stereo generation
/// - Can generate 30 seconds in a few seconds
/// </remarks>
public class MusicGenModel<T> : IAudioGenerationModel<T>
{
    private readonly MusicGenConfig _config;
    private readonly TextEncoder<T> _textEncoder;
    private readonly AudioLanguageModel<T> _audioLM;
    private readonly EnCodecDecoder<T> _decoder;

    public MusicGenModel(MusicGenConfig config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        // Text encoder: text → conditioning embeddings
        _textEncoder = new TextEncoder<T>(
            vocabSize: config.TextVocabSize,
            hiddenSize: config.HiddenSize);

        // Audio LM: predict audio codes given text condition
        _audioLM = new AudioLanguageModel<T>(
            numLayers: config.NumLayers,
            hiddenSize: config.HiddenSize,
            numCodebooks: config.NumCodebooks,
            codebookSize: config.CodebookSize);

        // Decoder: audio codes → waveform
        _decoder = new EnCodecDecoder<T>(
            numCodebooks: config.NumCodebooks,
            codebookSize: config.CodebookSize,
            sampleRate: config.SampleRate);
    }

    public AudioWaveform<T> Generate(
        string textPrompt,
        int durationSeconds = 10,
        double temperature = 1.0)
    {
        Guard.NotNullOrWhiteSpace(textPrompt, nameof(textPrompt));
        Guard.Positive(durationSeconds, nameof(durationSeconds));

        // Step 1: Encode text prompt
        var textEmbeddings = _textEncoder.Encode(textPrompt);

        // Step 2: Generate audio codes autoregressively
        int numFrames = durationSeconds * _config.FramesPerSecond;
        var audioCodes = _audioLM.Generate(
            conditionEmbeddings: textEmbeddings,
            numFrames: numFrames,
            temperature: temperature);

        // audioCodes shape: [num_codebooks, num_frames]

        // Step 3: Decode to waveform
        var waveform = _decoder.Decode(audioCodes);

        return waveform;
    }
}

public class MusicGenConfig
{
    public int NumLayers { get; set; } = 24;
    public int HiddenSize { get; set; } = 1024;
    public int NumCodebooks { get; set; } = 4;
    public int CodebookSize { get; set; } = 2048;
    public int TextVocabSize { get; set; } = 32000;
    public int SampleRate { get; set; } = 32000;
    public int FramesPerSecond { get; set; } = 50;  // 50 frames/sec
}
```

---

## Implementation Strategy

### Project Structure

```
src/
├── Audio/
│   ├── IAudioModel.cs
│   ├── AudioWaveform.cs
│   ├── Spectrogram.cs
│   ├── MelSpectrogram.cs
│   └── Preprocessing/
│       ├── AudioPreprocessor.cs
│       ├── SpectrogramExtractor.cs
│       ├── MelFilterbank.cs
│       └── AudioNormalizer.cs
│
├── Audio/Models/
│   ├── Wav2Vec2/
│   │   ├── Wav2Vec2Model.cs
│   │   ├── Wav2Vec2Config.cs
│   │   ├── CNNFeatureEncoder.cs
│   │   ├── CTCHead.cs
│   │   └── Wav2Vec2Processor.cs
│   │
│   ├── Whisper/
│   │   ├── WhisperModel.cs
│   │   ├── WhisperConfig.cs
│   │   ├── WhisperEncoder.cs
│   │   ├── WhisperDecoder.cs
│   │   └── WhisperProcessor.cs
│   │
│   └── MusicGen/
│       ├── MusicGenModel.cs
│       ├── MusicGenConfig.cs
│       ├── AudioLanguageModel.cs
│       ├── EnCodecDecoder.cs
│       └── MusicGenProcessor.cs
│
└── Audio/Utils/
    ├── FFT.cs (Fast Fourier Transform)
    ├── SignalProcessing.cs
    ├── AudioIO.cs (Load/save audio files)
    └── AudioAugmentation.cs (Data augmentation)
```

---

## Testing Strategy

### Unit Tests

```csharp
namespace AiDotNetTests.Audio;

public class AudioPreprocessorTests
{
    [Fact]
    public void Process_ValidWaveform_ReturnsMelSpectrogram()
    {
        // Arrange
        var preprocessor = new AudioPreprocessor<double>(
            targetSampleRate: 16000,
            melBins: 80);

        var waveform = CreateTestWaveform(sampleRate: 16000, durationSec: 1.0);

        // Act
        var melSpec = preprocessor.Process(waveform);

        // Assert
        Assert.NotNull(melSpec);
        Assert.Equal(80, melSpec.MelBins);
        Assert.True(melSpec.TimeFrames > 0);
    }

    [Fact]
    public void Resample_44100To16000_CorrectSampleCount()
    {
        // Arrange
        var preprocessor = new AudioPreprocessor<double>(targetSampleRate: 16000);
        var waveform = CreateTestWaveform(sampleRate: 44100, durationSec: 1.0);

        // Act
        var resampled = preprocessor.Resample(waveform, 16000);

        // Assert
        Assert.Equal(16000, resampled.SampleRate);
        Assert.Equal(16000, resampled.Samples);
    }

    [Fact]
    public void ConvertToMono_Stereo_AveragesChannels()
    {
        // Arrange
        var preprocessor = new AudioPreprocessor<double>();
        var stereo = new AudioWaveform<double>
        {
            Data = new Tensor<double>(new[] { 2, 100 }),  // 2 channels
            SampleRate = 16000,
            Channels = 2,
            Samples = 100
        };

        // Fill with test data
        for (int i = 0; i < 100; i++)
        {
            stereo.Data[0, i] = 1.0;  // Left channel
            stereo.Data[1, i] = -1.0; // Right channel
        }

        // Act
        var mono = preprocessor.ConvertToMono(stereo);

        // Assert
        Assert.Equal(1, mono.Channels);
        Assert.Equal(0.0, Convert.ToDouble(mono.Data[0, 0]), precision: 3);  // Average of 1 and -1
    }

    private AudioWaveform<double> CreateTestWaveform(int sampleRate, double durationSec)
    {
        int samples = (int)(sampleRate * durationSec);
        var data = new Tensor<double>(new[] { 1, samples });

        // Generate sine wave
        double frequency = 440.0;  // A4 note
        for (int i = 0; i < samples; i++)
        {
            double t = i / (double)sampleRate;
            data[0, i] = Math.Sin(2 * Math.PI * frequency * t);
        }

        return new AudioWaveform<double>
        {
            Data = data,
            SampleRate = sampleRate,
            Channels = 1,
            Samples = samples
        };
    }
}
```

### Integration Tests

```csharp
public class Wav2Vec2IntegrationTests
{
    [Fact]
    public void Wav2Vec2_ProcessAudio_ReturnsLogits()
    {
        // Arrange
        var config = new Wav2Vec2Config
        {
            HiddenSize = 768,
            NumLayers = 12,
            VocabSize = 32
        };
        var model = new Wav2Vec2Model<double>(config);

        // Create 1 second of audio
        var waveform = CreateTestWaveform(16000, 1.0);
        var batch = waveform.Data.Unsqueeze(0);  // Add batch dimension

        // Act
        var output = model.Forward(batch);

        // Assert
        Assert.NotNull(output.Logits);
        Assert.Equal(3, output.Logits.Rank);  // [batch, time, vocab]
        Assert.Equal(1, output.Logits.Shape[0]);  // Batch size
        Assert.Equal(32, output.Logits.Shape[2]);  // Vocab size
    }

    [Fact]
    public void Whisper_Encode_ReturnsHiddenStates()
    {
        // Arrange
        var config = new WhisperConfig();
        var model = new WhisperModel<double>(config);

        var preprocessor = new AudioPreprocessor<double>(
            targetSampleRate: 16000,
            melBins: 80);

        var waveform = CreateTestWaveform(16000, 10.0);  // 10 seconds
        var melSpec = preprocessor.Process(waveform);

        // Act
        var output = model.Forward(melSpec);

        // Assert
        Assert.NotNull(output.EncoderHiddenStates);
        Assert.NotNull(output.Logits);
        Assert.Equal(config.VocabSize, output.Logits.Shape[^1]);
    }
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Core Audio Infrastructure (6 hours)

#### AC 1.1: Audio Data Structures
**File**: `src/Audio/AudioWaveform.cs`

```csharp
namespace AiDotNet.Audio;

public class AudioWaveform<T>
{
    public Tensor<T> Data { get; set; } = new Tensor<T>(new[] { 1, 0 });
    public int SampleRate { get; set; }
    public int Channels { get; set; }
    public int Samples { get; set; }

    public double DurationSeconds => Samples / (double)SampleRate;

    public AudioWaveform<T> Clone()
    {
        return new AudioWaveform<T>
        {
            Data = Data.Clone(),
            SampleRate = SampleRate,
            Channels = Channels,
            Samples = Samples
        };
    }
}
```

**File**: `src/Audio/Spectrogram.cs`

```csharp
namespace AiDotNet.Audio;

public class Spectrogram<T>
{
    public Tensor<T> Data { get; set; } = new Tensor<T>(new[] { 0, 0 });
    public int FrequencyBins { get; set; }
    public int TimeFrames { get; set; }
    public int SampleRate { get; set; }
    public int HopLength { get; set; }
    public int WindowSize { get; set; }

    public double TimeResolution => HopLength / (double)SampleRate;
}

public class MelSpectrogram<T>
{
    public Tensor<T> Data { get; set; } = new Tensor<T>(new[] { 0, 0 });
    public int MelBins { get; set; }
    public int TimeFrames { get; set; }
    public int SampleRate { get; set; }
    public int HopLength { get; set; }
    public int WindowSize { get; set; }
    public double FrequencyMin { get; set; }
    public double FrequencyMax { get; set; }
}
```

#### AC 1.2: Audio Preprocessing
**File**: `src/Audio/Preprocessing/AudioPreprocessor.cs`

Implement the full preprocessor class shown in the Audio Processing Fundamentals section.

**Tests**: `tests/Audio/AudioPreprocessorTests.cs`

### Phase 2: Wav2Vec2 Implementation (8 hours)

#### AC 2.1: CNN Feature Encoder
**File**: `src/Audio/Models/Wav2Vec2/CNNFeatureEncoder.cs`

```csharp
namespace AiDotNet.Audio.Models.Wav2Vec2;

public class CNNFeatureEncoder<T>
{
    private readonly List<ConvLayer<T>> _convLayers;

    public CNNFeatureEncoder(
        int inputChannels,
        int outputChannels,
        int[] kernelSizes,
        int[] strides,
        string activations)
    {
        Guard.Positive(inputChannels, nameof(inputChannels));
        Guard.Positive(outputChannels, nameof(outputChannels));
        Guard.NotNull(kernelSizes, nameof(kernelSizes));
        Guard.NotNull(strides, nameof(strides));

        _convLayers = new List<ConvLayer<T>>();

        int channels = inputChannels;
        for (int i = 0; i < kernelSizes.Length; i++)
        {
            int nextChannels = (i == kernelSizes.Length - 1)
                ? outputChannels
                : outputChannels / 2;

            _convLayers.Add(new ConvLayer<T>(
                inChannels: channels,
                outChannels: nextChannels,
                kernelSize: kernelSizes[i],
                stride: strides[i],
                activation: activations));

            channels = nextChannels;
        }
    }

    public Tensor<T> Forward(Tensor<T> waveform)
    {
        Guard.NotNull(waveform, nameof(waveform));

        // waveform: [batch, samples]
        // Add channel dimension: [batch, 1, samples]
        var x = waveform.Unsqueeze(1);

        // Apply CNN layers
        foreach (var layer in _convLayers)
        {
            x = layer.Forward(x);
        }

        // Output: [batch, channels, time_steps]
        return x;
    }
}
```

#### AC 2.2: Wav2Vec2 Model
**File**: `src/Audio/Models/Wav2Vec2/Wav2Vec2Model.cs`

Implement the complete model shown in the Architecture Overview section.

#### AC 2.3: Tests
**File**: `tests/Audio/Models/Wav2Vec2/Wav2Vec2Tests.cs`

### Phase 3: Whisper Implementation (10 hours)

#### AC 3.1: Whisper Encoder
**File**: `src/Audio/Models/Whisper/WhisperEncoder.cs`

```csharp
namespace AiDotNet.Audio.Models.Whisper;

public class WhisperEncoder<T>
{
    private readonly Conv1dLayer<T> _conv1;
    private readonly Conv1dLayer<T> _conv2;
    private readonly PositionalEncoding<T> _posEncoding;
    private readonly List<TransformerEncoderBlock<T>> _layers;
    private readonly LayerNorm<T> _layerNorm;

    public WhisperEncoder(
        int numLayers,
        int hiddenSize,
        int numHeads,
        int melBins)
    {
        Guard.Positive(numLayers, nameof(numLayers));
        Guard.Positive(hiddenSize, nameof(hiddenSize));
        Guard.Positive(numHeads, nameof(numHeads));
        Guard.Positive(melBins, nameof(melBins));

        // Two conv layers to process mel spectrogram
        _conv1 = new Conv1dLayer<T>(
            inChannels: melBins,
            outChannels: hiddenSize,
            kernelSize: 3,
            stride: 1,
            padding: 1);

        _conv2 = new Conv1dLayer<T>(
            inChannels: hiddenSize,
            outChannels: hiddenSize,
            kernelSize: 3,
            stride: 2,  // Downsample by 2
            padding: 1);

        _posEncoding = new PositionalEncoding<T>(
            maxLen: 1500,  // Max time frames
            hiddenSize: hiddenSize);

        _layers = new List<TransformerEncoderBlock<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _layers.Add(new TransformerEncoderBlock<T>(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                intermediateSize: hiddenSize * 4,
                dropoutRate: 0.0));
        }

        _layerNorm = new LayerNorm<T>(hiddenSize);
    }

    public Tensor<T> Forward(Tensor<T> melSpectrogram)
    {
        Guard.NotNull(melSpectrogram, nameof(melSpectrogram));

        // Input: [batch, mel_bins, time_frames]
        var x = _conv1.Forward(melSpectrogram);
        x = GELU(x);

        x = _conv2.Forward(x);
        x = GELU(x);

        // Transpose to [batch, time_frames, hidden_size]
        x = x.Transpose(1, 2);

        // Add positional encoding
        x = _posEncoding.Forward(x);

        // Apply transformer blocks
        foreach (var layer in _layers)
        {
            x = layer.Forward(x);
        }

        // Final layer norm
        x = _layerNorm.Forward(x);

        return x;
    }

    private Tensor<T> GELU(Tensor<T> x)
    {
        // Gaussian Error Linear Unit activation
        var result = x.Clone();
        for (int i = 0; i < x.Size; i++)
        {
            double val = Convert.ToDouble(x.Data[i]);
            double gelu = 0.5 * val * (1 + Math.Tanh(
                Math.Sqrt(2 / Math.PI) * (val + 0.044715 * Math.Pow(val, 3))));
            result.Data[i] = (T)(object)gelu;
        }
        return result;
    }
}
```

#### AC 3.2: Whisper Decoder
**File**: `src/Audio/Models/Whisper/WhisperDecoder.cs`

Implement with cross-attention to encoder outputs.

#### AC 3.3: Complete Whisper Model
**File**: `src/Audio/Models/Whisper/WhisperModel.cs`

### Phase 4: MusicGen Implementation (12 hours)

#### AC 4.1: Audio Language Model
**File**: `src/Audio/Models/MusicGen/AudioLanguageModel.cs`

Implement autoregressive generation with multiple codebooks.

#### AC 4.2: EnCodec Decoder
**File**: `src/Audio/Models/MusicGen/EnCodecDecoder.cs`

Implement residual vector quantization decoder.

#### AC 4.3: Complete MusicGen
**File**: `src/Audio/Models/MusicGen/MusicGenModel.cs`

### Phase 5: Documentation and Examples (4 hours)

#### AC 5.1: XML Documentation
Add comprehensive XML comments to all public APIs.

#### AC 5.2: Usage Examples
Create example projects showing:
- Speech transcription with Whisper
- Speaker recognition with Wav2Vec2
- Music generation with MusicGen

---

## Checklist Summary

### Phase 1: Core Infrastructure (6 hours)
- [ ] Implement AudioWaveform, Spectrogram, MelSpectrogram classes
- [ ] Implement AudioPreprocessor with resampling
- [ ] Implement FFT and mel filterbank
- [ ] Write unit tests for preprocessing
- [ ] Test with real audio files

### Phase 2: Wav2Vec2 (8 hours)
- [ ] Implement CNN feature encoder
- [ ] Implement CTC head
- [ ] Integrate transformer encoder
- [ ] Create Wav2Vec2Model
- [ ] Write integration tests
- [ ] Test with speech data

### Phase 3: Whisper (10 hours)
- [ ] Implement Whisper encoder
- [ ] Implement Whisper decoder with cross-attention
- [ ] Create WhisperModel
- [ ] Implement beam search decoding
- [ ] Write integration tests
- [ ] Test multilingual transcription

### Phase 4: MusicGen (12 hours)
- [ ] Implement audio language model
- [ ] Implement EnCodec decoder
- [ ] Create MusicGenModel
- [ ] Implement autoregressive sampling
- [ ] Write integration tests
- [ ] Test music generation quality

### Phase 5: Documentation (4 hours)
- [ ] Add XML documentation to all classes
- [ ] Create usage examples
- [ ] Write performance benchmarks
- [ ] Document model configurations

### Total Estimated Time: 40 hours

---

## Success Criteria

1. **Preprocessing**: Audio correctly converted to spectrograms
2. **Wav2Vec2**: Achieves >90% accuracy on test speech data
3. **Whisper**: Transcribes multilingual audio with low WER
4. **MusicGen**: Generates coherent music matching text prompts
5. **Tests**: 80%+ coverage, all integration tests pass
6. **Performance**: Real-time or near real-time inference
7. **Documentation**: Complete XML docs and examples

---

## Common Pitfalls

### Pitfall 1: Incorrect Spectrogram Dimensions
**Problem**: Transposing frequency/time axes.
**Solution**: Always [frequency, time] for spectrograms.

### Pitfall 2: Sample Rate Mismatch
**Problem**: Model trained on 16kHz, input is 44.1kHz.
**Solution**: Always resample to model's expected rate.

### Pitfall 3: Forgetting Log Scaling
**Problem**: Raw spectrogram values too large.
**Solution**: Apply log scaling after mel conversion.

### Pitfall 4: Missing Normalization
**Problem**: Features have different scales.
**Solution**: Normalize to zero mean, unit variance.

---

## Resources

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [Audio Signal Processing Tutorial](https://jackschaedler.github.io/circles-sines-signals/)
- [Mel Spectrogram Explained](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
