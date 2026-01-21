---
layout: default
title: Audio Processing
parent: Tutorials
nav_order: 6
has_children: true
permalink: /tutorials/audio/
---

# Audio Processing Tutorial
{: .no_toc }

Learn to process audio with AiDotNet's speech and audio APIs.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What is Audio Processing?

Audio processing with machine learning includes:
- Speech-to-Text (transcription)
- Text-to-Speech (synthesis)
- Speaker recognition and diarization
- Audio classification
- Music generation and analysis

## AiDotNet Audio Capabilities

| Feature | Description | Model |
|:--------|:------------|:------|
| Transcription | Convert speech to text | Whisper |
| TTS | Convert text to speech | VITS, Tacotron2 |
| Speaker ID | Identify who is speaking | ECAPA-TDNN |
| Diarization | Segment by speaker | SpeakerDiarizer |
| Classification | Classify audio events | PANNs, AST |
| Enhancement | Remove noise | DeepFilterNet, DCCRN |

---

## Quick Start - Speech Transcription

```csharp
using AiDotNet.Audio;
using AiDotNet.Audio.Whisper;

// Load Whisper model
var whisper = new WhisperModel<float>(new WhisperOptions
{
    ModelSize = WhisperModelSize.Base,
    Language = "en"
});

// Transcribe audio file
var result = await whisper.TranscribeAsync("speech.wav");

Console.WriteLine($"Transcription: {result.Text}");

// Access segments with timestamps
foreach (var segment in result.Segments)
{
    Console.WriteLine($"[{segment.Start:F2}s - {segment.End:F2}s] {segment.Text}");
}
```

---

## Speech Recognition (Whisper)

### Model Sizes

| Size | Parameters | English | Multilingual | Speed |
|:-----|:-----------|:--------|:-------------|:------|
| `Tiny` | 39M | Good | Fair | Fastest |
| `Base` | 74M | Better | Good | Fast |
| `Small` | 244M | Great | Great | Medium |
| `Medium` | 769M | Excellent | Excellent | Slow |
| `Large` | 1.5B | Best | Best | Slowest |

### Basic Transcription

```csharp
var whisper = new WhisperModel<float>(new WhisperOptions
{
    ModelSize = WhisperModelSize.Small,
    Device = DeviceType.GPU  // Use GPU acceleration
});

// From file
var resultFromFile = await whisper.TranscribeAsync("audio.wav");

// From stream
using var stream = File.OpenRead("audio.wav");
var resultFromStream = await whisper.TranscribeAsync(stream);

// From byte array
var audioData = await File.ReadAllBytesAsync("audio.wav");
var resultFromBytes = await whisper.TranscribeAsync(audioData);
```

### Advanced Options

```csharp
var result = await whisper.TranscribeAsync("audio.wav", new TranscriptionOptions
{
    Language = "en",              // Force language (auto-detect if null)
    Task = WhisperTask.Transcribe,  // or WhisperTask.Translate
    WordTimestamps = true,        // Get word-level timestamps
    BeamSize = 5,                 // Beam search width
    Temperature = 0.0f,           // Sampling temperature (0 = greedy)
    VadFilter = true,             // Voice activity detection
    InitialPrompt = "Technical meeting about AI"  // Context hint
});

// Access word-level timestamps
foreach (var word in result.Words)
{
    Console.WriteLine($"{word.Word} [{word.Start:F2}s - {word.End:F2}s] (conf: {word.Probability:P0})");
}
```

### Real-time Streaming

```csharp
// Real-time transcription from microphone
var whisper = new WhisperModel<float>(new WhisperOptions
{
    ModelSize = WhisperModelSize.Base,
    StreamingMode = true
});

await foreach (var segment in whisper.TranscribeStreamAsync(microphoneStream))
{
    Console.Write(segment.Text);  // Stream output as it's transcribed
}
```

---

## Text-to-Speech (TTS)

### VITS Model (Recommended)

```csharp
using AiDotNet.Audio.TextToSpeech;

var tts = new VITSModel<float>(new TtsOptions
{
    Language = "en",
    SpeakerId = 0  // Multi-speaker models support different voices
});

// Generate speech
var audio = await tts.SynthesizeAsync("Hello, welcome to AiDotNet!");

// Save to file
await audio.SaveAsync("output.wav");

// Or get raw samples
var samples = audio.GetSamples();
```

### Tacotron2 Model

```csharp
var tts = new Tacotron2Model<float>(new TtsOptions
{
    VocoderType = VocoderType.HiFiGAN,
    SpeakingRate = 1.0f,
    Pitch = 1.0f
});

var audio = await tts.SynthesizeAsync("Converting text to natural speech.");
```

### Adjusting Voice

```csharp
// Speed up/slow down
audio = await tts.SynthesizeAsync("Fast speech", speakingRate: 1.5f);
audio = await tts.SynthesizeAsync("Slow speech", speakingRate: 0.75f);

// Adjust pitch
audio = await tts.SynthesizeAsync("Higher pitch", pitch: 1.2f);
audio = await tts.SynthesizeAsync("Lower pitch", pitch: 0.8f);

// Add pauses with SSML
var ssml = "<speak>Hello. <break time='500ms'/> How are you?</speak>";
audio = await tts.SynthesizeSsmlAsync(ssml);
```

---

## Speaker Recognition

### Speaker Verification

Verify if two audio samples are from the same speaker.

```csharp
using AiDotNet.Audio.Speaker;

var verifier = new SpeakerVerifier<float>(new SpeakerVerifierOptions
{
    Threshold = 0.5f  // Similarity threshold
});

// Compare two audio samples
var result = await verifier.VerifyAsync("sample1.wav", "sample2.wav");

Console.WriteLine($"Same speaker: {result.IsSameSpeaker}");
Console.WriteLine($"Similarity: {result.Similarity:F4}");
Console.WriteLine($"Confidence: {result.Confidence:P0}");
```

### Speaker Identification

Identify a speaker from a known set.

```csharp
var identifier = new SpeakerEmbeddingExtractor<float>();

// Enroll speakers
await identifier.EnrollAsync("alice", "alice_sample1.wav");
await identifier.EnrollAsync("alice", "alice_sample2.wav");  // Multiple samples improve accuracy
await identifier.EnrollAsync("bob", "bob_sample1.wav");
await identifier.EnrollAsync("charlie", "charlie_sample1.wav");

// Identify unknown speaker
var result = await identifier.IdentifyAsync("unknown.wav");

Console.WriteLine($"Identified: {result.SpeakerId}");
Console.WriteLine($"Confidence: {result.Confidence:P0}");

// Get rankings
foreach (var match in result.Matches.OrderByDescending(m => m.Score).Take(3))
{
    Console.WriteLine($"  {match.SpeakerId}: {match.Score:F4}");
}
```

### Speaker Diarization

Segment audio by speaker (who spoke when).

```csharp
var diarizer = new SpeakerDiarizer<float>(new SpeakerDiarizerOptions
{
    MinSpeakers = 2,
    MaxSpeakers = 10,
    MinSegmentDuration = 0.5  // Minimum segment length in seconds
});

var result = await diarizer.DiarizeAsync("meeting.wav");

Console.WriteLine($"Detected {result.NumSpeakers} speakers");
Console.WriteLine();

foreach (var turn in result.Turns)
{
    Console.WriteLine($"[{turn.Start:F2}s - {turn.End:F2}s] Speaker {turn.SpeakerId}");
}
```

---

## Audio Classification

### Event Detection

```csharp
using AiDotNet.Audio.Classification;

var detector = new AudioEventDetector<float>(new AudioEventDetectorOptions
{
    MinConfidence = 0.5f
});

var events = await detector.DetectAsync("audio.wav");

foreach (var evt in events)
{
    Console.WriteLine($"[{evt.Start:F2}s] {evt.Label} (confidence: {evt.Confidence:P0})");
}
// Example output:
// [0.00s] Speech (confidence: 95%)
// [3.50s] Dog bark (confidence: 87%)
// [5.20s] Door slam (confidence: 72%)
```

### Music Genre Classification

```csharp
var classifier = new GenreClassifier<float>();

var result = await classifier.ClassifyAsync("song.mp3");

Console.WriteLine($"Genre: {result.PredictedGenre}");
Console.WriteLine($"Confidence: {result.Confidence:P0}");

// Get all predictions
foreach (var (genre, prob) in result.AllProbabilities.OrderByDescending(p => p.Value).Take(5))
{
    Console.WriteLine($"  {genre}: {prob:P1}");
}
```

### Scene Classification

```csharp
var sceneClassifier = new SceneClassifier<float>();

var result = await sceneClassifier.ClassifyAsync("recording.wav");

Console.WriteLine($"Scene: {result.PredictedScene}");  // e.g., "office", "street", "park"
```

---

## Audio Enhancement

### Noise Reduction

```csharp
using AiDotNet.Audio.Enhancement;

var denoiser = new DeepFilterNet<float>(new DeepFilterNetOptions
{
    AttenLimit = 100,  // Maximum noise attenuation in dB
    PostFilter = true
});

var cleanAudio = await denoiser.EnhanceAsync("noisy_speech.wav");
await cleanAudio.SaveAsync("clean_speech.wav");
```

### Speech Enhancement

```csharp
var enhancer = new DCCRN<float>();

var enhanced = await enhancer.EnhanceAsync("poor_quality.wav");
await enhanced.SaveAsync("enhanced.wav");
```

---

## Audio Feature Extraction

### Mel-Frequency Cepstral Coefficients (MFCCs)

```csharp
using AiDotNet.Audio.Features;

var mfccExtractor = new MfccExtractor<float>(new MfccOptions
{
    NumCoefficients = 13,
    SampleRate = 16000,
    FrameLength = 0.025f,  // 25ms
    FrameStep = 0.010f     // 10ms
});

var mfccs = mfccExtractor.Extract("audio.wav");
// Shape: [numFrames, numCoefficients]
```

### Spectrogram

```csharp
var spectralExtractor = new SpectralFeatureExtractor<float>(new SpectralFeatureOptions
{
    FeatureType = SpectralFeatureType.MelSpectrogram,
    NumMelBins = 80,
    WindowType = WindowType.Hann
});

var spectrogram = spectralExtractor.Extract("audio.wav");
```

### Chroma Features

```csharp
var chromaExtractor = new ChromaExtractor<float>(new ChromaOptions
{
    NumChroma = 12,
    Tuning = 440.0f  // A4 = 440 Hz
});

var chroma = chromaExtractor.Extract("music.wav");
```

---

## Music Analysis

### Beat Tracking

```csharp
using AiDotNet.Audio.MusicAnalysis;

var beatTracker = new BeatTracker<float>();

var result = await beatTracker.TrackAsync("song.mp3");

Console.WriteLine($"BPM: {result.Tempo:F1}");

foreach (var beat in result.Beats)
{
    Console.WriteLine($"Beat at {beat:F3}s");
}
```

### Key Detection

```csharp
var keyDetector = new KeyDetector<float>();

var result = await keyDetector.DetectAsync("song.mp3");

Console.WriteLine($"Key: {result.Key} {result.Mode}");  // e.g., "C Major"
Console.WriteLine($"Confidence: {result.Confidence:P0}");
```

### Chord Recognition

```csharp
var chordRecognizer = new ChordRecognizer<float>();

var result = await chordRecognizer.RecognizeAsync("song.mp3");

foreach (var segment in result.Segments)
{
    Console.WriteLine($"[{segment.Start:F2}s - {segment.End:F2}s] {segment.Chord}");
}
```

---

## Audio Formats and I/O

### Supported Formats

- WAV (PCM, float)
- MP3
- FLAC
- OGG
- M4A/AAC

### Loading Audio

```csharp
// From file
var audio = AudioData.Load("input.wav");

// From URL
var audio = await AudioData.LoadFromUrlAsync("https://example.com/audio.wav");

// From stream
using var stream = File.OpenRead("input.wav");
var audio = AudioData.Load(stream);
```

### Resampling

```csharp
// Resample to 16kHz (common for speech models)
var resampled = audio.Resample(targetSampleRate: 16000);

// Convert to mono
var mono = audio.ToMono();

// Normalize volume
var normalized = audio.Normalize();
```

---

## Best Practices

1. **Sample Rate**: Most speech models expect 16kHz audio
2. **Mono Audio**: Convert stereo to mono for most speech tasks
3. **Preprocessing**: Normalize audio and remove silence for better results
4. **GPU Acceleration**: Use GPU for large files or real-time processing
5. **Batching**: Process multiple files in batches for efficiency

---

## Common Issues

### Poor Transcription Quality

- Ensure audio is clear with minimal background noise
- Use larger Whisper models for difficult audio
- Consider preprocessing with noise reduction

### Slow Processing

- Use smaller models for real-time applications
- Enable GPU acceleration
- Use streaming mode for long audio files

### Memory Issues

- Process long audio in chunks
- Use streaming APIs
- Reduce model size if memory-constrained

---

## Next Steps

- [Speech Recognition Sample](../../../samples/audio/SpeechRecognition/)
- [Text-to-Speech Sample](../../../samples/audio/TextToSpeech/)
- [Audio API Reference](../../api/)
