# Audio Processing Samples

This directory contains examples of audio processing models in AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [Whisper](./SpeechRecognition/Whisper/) | Speech-to-text using Whisper |
| [TextToSpeech](./TextToSpeech/) | Convert text to natural speech |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Audio;

// Speech Recognition
var whisper = new WhisperModel<float>(modelSize: WhisperSize.Base);
var transcript = await whisper.TranscribeAsync("audio.wav");

// Text-to-Speech
var tts = new TextToSpeechModel<float>();
var audio = await tts.SynthesizeAsync("Hello, world!");
```

## Audio Models (90+)

### Speech Recognition
- Whisper (tiny, base, small, medium, large)
- Wav2Vec2
- Conformer

### Text-to-Speech
- VITS
- Tacotron2
- FastSpeech2

### Audio Classification
- AudioSpectrogramTransformer
- PANNs
- VGGish

### Music
- MusicGen
- Jukebox
- AudioCraft

### Enhancement
- Demucs (source separation)
- RNNoise (noise reduction)
- SpeechBrain

## Learn More

- [Audio Tutorial](/docs/tutorials/audio/)
- [API Reference](/api/AiDotNet.Audio/)
