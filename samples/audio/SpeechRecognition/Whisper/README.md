# Speech Recognition with Whisper

This sample demonstrates automatic speech recognition (ASR) using OpenAI's Whisper model.

## What You'll Learn

- How to configure Whisper for speech-to-text
- How to process audio files
- How to get timestamps and language detection
- How to handle different audio formats

## What is Whisper?

Whisper is an automatic speech recognition model trained on 680,000 hours of multilingual audio. It can:
- Transcribe audio to text
- Translate to English
- Detect the language
- Generate timestamps

## Available Models

| Model | Parameters | Speed | Accuracy | VRAM |
|-------|------------|-------|----------|------|
| tiny | 39M | Fastest | Good | ~1GB |
| base | 74M | Fast | Better | ~1GB |
| small | 244M | Medium | Good+ | ~2GB |
| medium | 769M | Slower | Great | ~5GB |
| large-v3 | 1.5B | Slowest | Best | ~10GB |

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
=== AiDotNet Whisper Speech Recognition ===

Loading Whisper model (base)...
  Model loaded successfully

Processing audio file: sample.wav
  Duration: 10.5 seconds
  Sample rate: 16000 Hz

Transcribing...
  Language detected: English (confidence: 0.98)

Transcription:
─────────────────────────────────────
Hello, this is a test of the Whisper speech recognition system.
It can transcribe audio in multiple languages and generate timestamps.

Word-level timestamps:
  [0.00 - 0.50] Hello,
  [0.50 - 1.20] this
  [1.20 - 1.50] is
  [1.50 - 1.80] a
  [1.80 - 2.20] test
  ...
```

## Code Highlights

```csharp
// Load Whisper model
var whisper = new WhisperModel<float>(
    modelSize: WhisperModelSize.Base,
    language: "en",  // null for auto-detect
    device: DeviceType.GPU);

// Transcribe audio file
var result = await whisper.TranscribeAsync("audio.wav", new TranscriptionOptions
{
    ReturnTimestamps = true,
    WordTimestamps = true,
    LanguageDetection = true
});

Console.WriteLine(result.Text);
```

## Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

## Tips for Best Results

1. **Audio quality**: Use 16kHz mono audio for best results
2. **Background noise**: Clean audio transcribes better
3. **Model size**: Larger models are more accurate but slower
4. **GPU**: Use GPU acceleration for faster processing

## Next Steps

- [TextToSpeech](../../TextToSpeech/) - Generate speech from text
- [AudioClassification](../../AudioClassification/) - Classify audio events
