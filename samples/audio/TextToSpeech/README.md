# Text-to-Speech with VITS

This sample demonstrates neural text-to-speech (TTS) synthesis using VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech).

## What You'll Learn

- How to configure and use neural TTS models
- How to select different voices
- How to control speech speed and pitch
- How to add emotion/style to synthesized speech
- How to use SSML for advanced control
- How to analyze phoneme-level timing
- How to batch synthesize multiple texts
- How to stream audio for real-time applications

## What is VITS?

VITS is a state-of-the-art text-to-speech model that combines:
- **Variational Autoencoder (VAE)** for learning latent representations
- **Normalizing Flows** for improved expressiveness
- **Adversarial Training** for high-quality audio generation

Key advantages:
- End-to-end: Text directly to waveform (no separate vocoder needed)
- Parallel synthesis: Fast inference compared to autoregressive models
- High quality: Natural, human-like speech
- Multi-speaker: Single model supports multiple voices

## Available TTS Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| VITS | Fast | Excellent | General purpose, production |
| FastSpeech2 | Very Fast | Good | Real-time applications |
| Tacotron2 | Slow | Great | High-quality offline synthesis |
| GlowTTS | Fast | Very Good | Diverse speech patterns |
| StyleTTS2 | Medium | Excellent | Style-controllable speech |

## Running the Sample

```bash
cd samples/audio/TextToSpeech
dotnet run
```

## Expected Output

```
=== AiDotNet Text-to-Speech Generation ===
Neural TTS with VITS (Variational Inference Text-to-Speech)

Available TTS Models:
  VITS           - High-quality, end-to-end TTS
  FastSpeech2    - Fast, parallel synthesis
  Tacotron2      - Attention-based seq2seq
  GlowTTS        - Flow-based for diverse speech
  StyleTTS2      - Style-controllable TTS

Configuration:
  Model: VITS
  Language: en
  Sample Rate: 22050 Hz
  GPU Acceleration: True

Loading TTS model...
  Model loaded successfully

Available Voices:
----------------------------------------------------------------------
| ID  | Name           | Gender | Style      | Language |
----------------------------------------------------------------------
| 1   | James          | Male   | Neutral    | en-US    |
| 2   | Michael        | Male   | Warm       | en-US    |
| 3   | Sarah          | Female | Neutral    | en-US    |
| 4   | Emily          | Female | Bright     | en-US    |
----------------------------------------------------------------------

=== Demo 1: Basic Text-to-Speech ===

Text: "Hello! Welcome to AiDotNet's text-to-speech demonstration..."

Synthesis Result:
  Duration: 4.20 seconds
  Sample Rate: 22050 Hz
  Samples: 92,610
  Synthesis Time: 45.23ms
  Real-time Factor: 92.8x

Audio Waveform (first 100ms):
  +------------------------------------------------------------+
  |    *   *   *    *  *   *   *    *   *   *    *   *   *     | +1
  |   * * * * * *  * ** * * * * *  * * * * * *  * * * * * *    |
  |  *   *   *   **    *   *   *  *   *   *   **   *   *   *   | 0
  |                                                             |
  +------------------------------------------------------------+ -1
   0ms                                                      100ms

=== Demo 3: Speed Control ===

| Speed  | Duration | RTF    | Description        |
-------------------------------------------------------
|  0.75x |    5.60s | 85.2x  | Slow, deliberate   |
|  1.00x |    4.20s | 92.8x  | Normal speed       |
|  1.25x |    3.36s | 98.5x  | Faster paced       |
|  1.50x |    2.80s | 102.3x | Quick speech       |
-------------------------------------------------------

=== Demo 5: Emotion/Style Control ===

| Emotion  | Energy | Pitch Var | Speaking Rate |
--------------------------------------------------
| Neutral  |  0.60  |      0.20 |          1.00 |
| Happy    |  0.75  |      0.35 |          1.15 |
| Sad      |  0.45  |      0.15 |          0.85 |
| Excited  |  0.90  |      0.45 |          1.25 |
--------------------------------------------------
```

## Code Highlights

### Basic Text-to-Speech

```csharp
// Configure TTS
var config = new TTSConfig
{
    Model = TTSModelType.VITS,
    Language = "en",
    SampleRate = 22050
};

// Create synthesizer
var tts = new TextToSpeechSynthesizer(config);

// Synthesize text
var result = tts.Synthesize("Hello, world!");

// Save to file
tts.SaveToWav(result, "output.wav");
```

### Voice Selection

```csharp
// Get available voices
var voices = tts.GetAvailableVoices();

// Select a specific voice
var options = new SynthesisOptions
{
    VoiceId = "en_female_2"
};

var result = tts.Synthesize("Hello!", options);
```

### Speed and Pitch Control

```csharp
var options = new SynthesisOptions
{
    Speed = 1.2f,   // 20% faster
    Pitch = 0.9f    // Slightly lower pitch
};

var result = tts.Synthesize(text, options);
```

### Emotion/Style Control

```csharp
var options = new SynthesisOptions
{
    VoiceId = "en_female_1",
    Emotion = SpeechEmotion.Happy
};

var result = tts.Synthesize("Great news!", options);
```

### SSML Support

```csharp
var ssml = @"<speak>
    <s>Welcome to our service.</s>
    <s><emphasis level=""strong"">Important announcement:</emphasis></s>
    <s>Please wait<break time=""500ms""/>for further instructions.</s>
</speak>";

var result = tts.SynthesizeSsml(ssml);
```

### Real-time Streaming

```csharp
// Stream synthesis for real-time playback
await foreach (var chunk in tts.SynthesizeStreamAsync(longText))
{
    audioPlayer.EnqueueChunk(chunk.AudioData);
}
```

## Voice Characteristics

### Voice Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Speed | 0.5 - 2.0 | Speaking rate |
| Pitch | 0.5 - 2.0 | Voice frequency |
| Volume | 0.0 - 1.0 | Output loudness |
| Energy | 0.0 - 1.0 | Voice intensity |

### Emotion Presets

| Emotion | Energy | Pitch Var | Rate | Best For |
|---------|--------|-----------|------|----------|
| Neutral | Medium | Low | Normal | General content |
| Happy | High | Medium | Faster | Positive messages |
| Sad | Low | Low | Slower | Empathetic content |
| Excited | Very High | High | Fast | Announcements |
| Calm | Low | Very Low | Slow | Relaxation, meditation |

## SSML Reference

### Basic Tags

```xml
<!-- Sentence boundary -->
<s>This is a sentence.</s>

<!-- Paragraph boundary -->
<p>This is a paragraph.</p>

<!-- Pause -->
<break time="500ms"/>

<!-- Emphasis -->
<emphasis level="strong">Important</emphasis>
```

### Pronunciation Control

```xml
<!-- Say as specific type -->
<say-as interpret-as="cardinal">42</say-as>
<say-as interpret-as="date">2024-01-15</say-as>
<say-as interpret-as="telephone">+1-800-555-1234</say-as>

<!-- Custom phonemes -->
<phoneme alphabet="ipa" ph="tuh-MEY-toh">tomato</phoneme>
```

### Prosody Control

```xml
<!-- Rate, pitch, and volume -->
<prosody rate="fast" pitch="high" volume="loud">
    Exciting news!
</prosody>
```

## Performance Optimization

### GPU Acceleration

```csharp
var config = new TTSConfig
{
    UseGpu = true  // Enable CUDA acceleration
};
```

### Batch Processing

```csharp
// Process multiple texts efficiently
var texts = new[] { "First", "Second", "Third" };
var results = tts.SynthesizeBatch(texts);
```

### Caching

```csharp
// Enable synthesis caching for repeated phrases
var config = new TTSConfig
{
    EnableCache = true,
    CacheSize = 1000  // Cache up to 1000 phrases
};
```

## Audio Export Formats

| Format | Compression | Quality | Use Case |
|--------|-------------|---------|----------|
| WAV | None | Lossless | Editing, archival |
| MP3 | Lossy | Good | Web, mobile |
| OGG | Lossy | Good | Gaming, web |
| FLAC | Lossless | Perfect | High-quality archival |

## Architecture

```
Text Input
    |
    v
+------------------+
| Text Encoder     |  Transformer layers for text processing
| (Transformer)    |  Learns text-to-phoneme alignment
+------------------+
    |
    v
+------------------+
| Duration         |  Predicts phoneme durations
| Predictor        |  Uses variational inference
+------------------+
    |
    v
+------------------+
| Flow-based       |  Normalizing flow network
| Decoder          |  Converts latent to mel features
+------------------+
    |
    v
+------------------+
| HiFi-GAN         |  Neural vocoder
| Vocoder          |  Mel spectrogram to waveform
+------------------+
    |
    v
Audio Waveform (22050 Hz)
```

## Tips for Best Results

1. **Text preprocessing**: Clean up abbreviations and numbers
2. **Punctuation**: Use proper punctuation for natural pauses
3. **SSML**: Use for precise control over pronunciation
4. **Voice selection**: Match voice to content type
5. **Speed**: Adjust based on content complexity

## Next Steps

- [SpeechRecognition/Whisper](../SpeechRecognition/Whisper/) - Convert speech back to text
- [AudioClassification](../AudioClassification/) - Classify audio events
- [VoiceCloning](../VoiceCloning/) - Clone custom voices

## Resources

- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [SSML Specification](https://www.w3.org/TR/speech-synthesis11/)
- [HiFi-GAN Paper](https://arxiv.org/abs/2010.05646)
