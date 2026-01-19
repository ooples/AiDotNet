# Speech Assistant Sample

A complete voice assistant application powered by AiDotNet's Whisper speech recognition and Text-to-Speech models.

## Features

- **Whisper Speech Recognition** - State-of-the-art speech-to-text
- **Text-to-Speech Synthesis** - Multiple voice options
- **In-Browser Recording** - Record directly from microphone
- **Audio File Upload** - Support for WAV, MP3, OGG
- **Word-Level Timestamps** - Precise transcription segments
- **Interactive Web UI** - Modern interface for all features

## Quick Start

```bash
cd samples/end-to-end/SpeechAssistant
dotnet run
```

Then open http://localhost:5001 in your browser.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web UI                               │
├─────────────────────────────────────────────────────────────┤
│                     REST API Layer                          │
├──────────────────────────┬──────────────────────────────────┤
│    Speech-to-Text        │       Text-to-Speech             │
│    (Whisper Model)       │       (TTS Model)                │
├──────────────────────────┼──────────────────────────────────┤
│    Audio Decoding        │       Audio Encoding             │
│    (WAV, MP3, OGG)       │       (WAV output)               │
└──────────────────────────┴──────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/transcribe` | POST | Upload audio file for transcription |
| `/api/transcribe/url` | POST | Transcribe audio from URL |
| `/api/synthesize` | POST | Convert text to speech |
| `/api/voices` | GET | List available TTS voices |
| `/api/stats` | GET | Get usage statistics |

## Example API Usage

### Transcribe Audio File

```bash
curl -X POST http://localhost:5001/api/transcribe \
  -F "audio=@recording.wav"
```

Response:
```json
{
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "duration": 3.5,
  "processingTime": 1.2,
  "segments": [
    { "start": 0.0, "end": 1.5, "text": "Hello,", "confidence": 0.95 },
    { "start": 1.5, "end": 3.5, "text": "this is a test transcription.", "confidence": 0.92 }
  ]
}
```

### Transcribe from URL

```bash
curl -X POST http://localhost:5001/api/transcribe/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/audio.wav"}'
```

### Text-to-Speech

```bash
curl -X POST http://localhost:5001/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "alloy", "speed": 1.0}' \
  --output speech.wav
```

## Available Voices

| Voice | Description |
|-------|-------------|
| alloy | Neutral, balanced voice |
| echo | Warm, friendly voice |
| fable | Expressive, storytelling voice |
| onyx | Deep, authoritative voice |
| nova | Youthful, energetic voice |
| shimmer | Soft, gentle voice |

## Configuration

### Whisper Model Options

```csharp
// Choose model size based on accuracy/speed tradeoff
_whisper = await WhisperModel<float>.LoadAsync(
    model: "whisper-base",  // tiny, base, small, medium, large
    language: "en");        // or "auto" for detection
```

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39M | Fastest | Good |
| base | 74M | Fast | Better |
| small | 244M | Medium | Great |
| medium | 769M | Slow | Excellent |
| large | 1.5B | Slowest | Best |

### TTS Configuration

```csharp
var config = new TTSConfig
{
    Voice = "alloy",
    Speed = 1.0f,        // 0.5 to 2.0
    Format = AudioFormat.WAV,
    SampleRate = 24000
};
```

## Web UI Features

### Speech-to-Text Tab
- Click the microphone to record from browser
- Upload audio files via drag & drop
- View transcription with timestamps
- See processing statistics

### Text-to-Speech Tab
- Enter text to synthesize
- Choose from 6 different voices
- Adjust speech speed (0.5x - 2.0x)
- Play synthesized audio in browser

### Statistics Tab
- Total transcriptions processed
- Total syntheses generated
- Audio duration processed
- Model information

## Supported Audio Formats

- WAV (recommended)
- MP3
- OGG
- FLAC
- M4A
- WebM

## Requirements

- .NET 8.0 SDK
- AiDotNet NuGet package
- Modern web browser (for recording)

## Learn More

- [Audio Processing Tutorial](https://ooples.github.io/AiDotNet/tutorials/audio/)
- [Whisper Sample](/samples/audio/SpeechRecognition/Whisper/)
- [TTS Sample](/samples/audio/TextToSpeech/)
- [Audio API Reference](https://ooples.github.io/AiDotNet/api/AiDotNet.Audio/)
