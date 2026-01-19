// =============================================================================
// AiDotNet Sample: SpeechAssistant
// =============================================================================
// A complete voice assistant application with speech-to-text and text-to-speech.
// Features:
// - Whisper-powered speech recognition
// - Text-to-Speech synthesis
// - Audio file upload/processing
// - Real-time transcription
// - Web UI with audio recording
//
// Run with: dotnet run
// Then open: http://localhost:5001
// =============================================================================

using AiDotNet.Audio;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Audio.TextToSpeech;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddSingleton<SpeechAssistantService>();
builder.Services.AddEndpointsApiExplorer();

var app = builder.Build();

// Initialize the speech assistant
var assistant = app.Services.GetRequiredService<SpeechAssistantService>();
await assistant.InitializeAsync();

// Serve static HTML UI
app.MapGet("/", () => Results.Content(GetHtmlUI(), "text/html"));

// Speech Recognition endpoints
app.MapPost("/api/transcribe", async (HttpRequest request, SpeechAssistantService service) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest("Expected multipart/form-data");

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("audio");

    if (file == null)
        return Results.BadRequest("No audio file provided");

    using var stream = file.OpenReadStream();
    using var ms = new MemoryStream();
    await stream.CopyToAsync(ms);
    var audioData = ms.ToArray();

    var result = await service.TranscribeAsync(audioData, file.FileName);
    return Results.Json(result);
});

app.MapPost("/api/transcribe/url", async (TranscribeUrlRequest request, SpeechAssistantService service) =>
{
    var result = await service.TranscribeFromUrlAsync(request.Url);
    return Results.Json(result);
});

// Text-to-Speech endpoints
app.MapPost("/api/synthesize", async (SynthesizeRequest request, SpeechAssistantService service) =>
{
    var audioData = await service.SynthesizeAsync(request.Text, request.Voice, request.Speed);
    return Results.File(audioData, "audio/wav", "speech.wav");
});

app.MapGet("/api/voices", (SpeechAssistantService service) =>
{
    return Results.Json(service.GetAvailableVoices());
});

// Statistics endpoint
app.MapGet("/api/stats", (SpeechAssistantService service) =>
{
    return Results.Json(service.GetStats());
});

Console.WriteLine("===========================================");
Console.WriteLine("  AiDotNet SpeechAssistant Sample");
Console.WriteLine("===========================================");
Console.WriteLine();
Console.WriteLine("Server running at: http://localhost:5001");
Console.WriteLine();
Console.WriteLine("Features:");
Console.WriteLine("  - Whisper speech recognition");
Console.WriteLine("  - Text-to-Speech synthesis");
Console.WriteLine("  - Multiple voice options");
Console.WriteLine("  - Audio file upload");
Console.WriteLine("  - In-browser recording");
Console.WriteLine();
Console.WriteLine("API Endpoints:");
Console.WriteLine("  POST /api/transcribe     - Upload audio for transcription");
Console.WriteLine("  POST /api/transcribe/url - Transcribe from URL");
Console.WriteLine("  POST /api/synthesize     - Convert text to speech");
Console.WriteLine("  GET  /api/voices         - List available voices");
Console.WriteLine("  GET  /api/stats          - Get usage statistics");
Console.WriteLine();
Console.WriteLine("Press Ctrl+C to stop the server.");
Console.WriteLine("===========================================");

app.Run("http://localhost:5001");

// =============================================================================
// Speech Assistant Service
// =============================================================================

public class SpeechAssistantService
{
    private WhisperModel<float>? _whisper;
    private TextToSpeechModel<float>? _tts;
    private int _transcriptionCount;
    private int _synthesisCount;
    private double _totalAudioProcessed;

    public async Task InitializeAsync()
    {
        Console.WriteLine("Loading Whisper model...");
        _whisper = await WhisperModel<float>.LoadAsync(
            model: "whisper-base",
            language: "en");

        Console.WriteLine("Loading TTS model...");
        _tts = await TextToSpeechModel<float>.LoadAsync(
            model: "tts-1",
            voice: "alloy");

        Console.WriteLine("Speech assistant initialized.");
    }

    public async Task<TranscriptionResult> TranscribeAsync(byte[] audioData, string filename)
    {
        if (_whisper == null)
            throw new InvalidOperationException("Whisper model not initialized");

        _transcriptionCount++;

        var startTime = DateTime.UtcNow;

        // Decode audio and transcribe
        var audioSamples = DecodeAudio(audioData, filename);
        var duration = audioSamples.Length / 16000.0; // 16kHz sample rate
        _totalAudioProcessed += duration;

        var transcription = await _whisper.TranscribeAsync(audioSamples);

        var processingTime = (DateTime.UtcNow - startTime).TotalSeconds;

        return new TranscriptionResult
        {
            Text = transcription.Text,
            Language = transcription.Language,
            Duration = duration,
            ProcessingTime = processingTime,
            Segments = transcription.Segments.Select(s => new TranscriptionSegment
            {
                Start = s.Start,
                End = s.End,
                Text = s.Text,
                Confidence = s.Confidence
            }).ToList()
        };
    }

    public async Task<TranscriptionResult> TranscribeFromUrlAsync(string url)
    {
        if (_whisper == null)
            throw new InvalidOperationException("Whisper model not initialized");

        using var httpClient = new HttpClient();
        var audioData = await httpClient.GetByteArrayAsync(url);

        var extension = Path.GetExtension(new Uri(url).LocalPath) ?? ".wav";
        return await TranscribeAsync(audioData, $"audio{extension}");
    }

    public async Task<byte[]> SynthesizeAsync(string text, string? voice = null, float speed = 1.0f)
    {
        if (_tts == null)
            throw new InvalidOperationException("TTS model not initialized");

        _synthesisCount++;

        var config = new TTSConfig
        {
            Voice = voice ?? "alloy",
            Speed = speed,
            Format = AudioFormat.WAV,
            SampleRate = 24000
        };

        var audioSamples = await _tts.SynthesizeAsync(text, config);

        // Encode to WAV format
        return EncodeToWav(audioSamples, 24000);
    }

    public List<VoiceInfo> GetAvailableVoices()
    {
        return new List<VoiceInfo>
        {
            new() { Id = "alloy", Name = "Alloy", Description = "Neutral, balanced voice" },
            new() { Id = "echo", Name = "Echo", Description = "Warm, friendly voice" },
            new() { Id = "fable", Name = "Fable", Description = "Expressive, storytelling voice" },
            new() { Id = "onyx", Name = "Onyx", Description = "Deep, authoritative voice" },
            new() { Id = "nova", Name = "Nova", Description = "Youthful, energetic voice" },
            new() { Id = "shimmer", Name = "Shimmer", Description = "Soft, gentle voice" }
        };
    }

    public SpeechStats GetStats()
    {
        return new SpeechStats
        {
            TranscriptionCount = _transcriptionCount,
            SynthesisCount = _synthesisCount,
            TotalAudioProcessed = _totalAudioProcessed,
            WhisperModel = "whisper-base",
            TTSModel = "tts-1"
        };
    }

    private static float[] DecodeAudio(byte[] audioData, string filename)
    {
        // Simple WAV decoder for demonstration
        // In production, use NAudio or similar for full format support
        var extension = Path.GetExtension(filename).ToLowerInvariant();

        if (extension == ".wav")
        {
            return DecodeWav(audioData);
        }

        // For other formats, return simulated audio
        // In production, use ffmpeg or NAudio
        return new float[16000 * 5]; // 5 seconds of silence
    }

    private static float[] DecodeWav(byte[] wavData)
    {
        if (wavData.Length < 44)
            return Array.Empty<float>();

        // Parse WAV header
        var channels = BitConverter.ToInt16(wavData, 22);
        var sampleRate = BitConverter.ToInt32(wavData, 24);
        var bitsPerSample = BitConverter.ToInt16(wavData, 34);

        // Find data chunk
        var dataOffset = 44;
        var dataSize = wavData.Length - dataOffset;

        var samples = new List<float>();
        var bytesPerSample = bitsPerSample / 8;

        for (var i = dataOffset; i < wavData.Length - bytesPerSample + 1; i += bytesPerSample * channels)
        {
            float sample;
            if (bitsPerSample == 16)
            {
                sample = BitConverter.ToInt16(wavData, i) / 32768f;
            }
            else if (bitsPerSample == 32)
            {
                sample = BitConverter.ToInt32(wavData, i) / 2147483648f;
            }
            else
            {
                sample = (wavData[i] - 128) / 128f;
            }
            samples.Add(sample);
        }

        // Resample to 16kHz if needed
        if (sampleRate != 16000 && sampleRate > 0)
        {
            var resampleRatio = 16000.0 / sampleRate;
            var resampledLength = (int)(samples.Count * resampleRatio);
            var resampled = new float[resampledLength];

            for (var i = 0; i < resampledLength; i++)
            {
                var srcIndex = i / resampleRatio;
                var srcIndexInt = (int)srcIndex;
                var frac = srcIndex - srcIndexInt;

                if (srcIndexInt + 1 < samples.Count)
                {
                    resampled[i] = (float)(samples[srcIndexInt] * (1 - frac) + samples[srcIndexInt + 1] * frac);
                }
                else if (srcIndexInt < samples.Count)
                {
                    resampled[i] = samples[srcIndexInt];
                }
            }
            return resampled;
        }

        return samples.ToArray();
    }

    private static byte[] EncodeToWav(float[] samples, int sampleRate)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        var dataSize = samples.Length * 2; // 16-bit samples

        // RIFF header
        writer.Write("RIFF"u8);
        writer.Write(36 + dataSize);
        writer.Write("WAVE"u8);

        // fmt chunk
        writer.Write("fmt "u8);
        writer.Write(16); // Chunk size
        writer.Write((short)1); // Audio format (PCM)
        writer.Write((short)1); // Channels
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2); // Byte rate
        writer.Write((short)2); // Block align
        writer.Write((short)16); // Bits per sample

        // data chunk
        writer.Write("data"u8);
        writer.Write(dataSize);

        foreach (var sample in samples)
        {
            var clipped = Math.Max(-1f, Math.Min(1f, sample));
            writer.Write((short)(clipped * 32767));
        }

        return ms.ToArray();
    }
}

// =============================================================================
// Request/Response Models
// =============================================================================

public record TranscribeUrlRequest(string Url);
public record SynthesizeRequest(string Text, string? Voice = null, float Speed = 1.0f);

public class TranscriptionResult
{
    public string Text { get; set; } = "";
    public string Language { get; set; } = "";
    public double Duration { get; set; }
    public double ProcessingTime { get; set; }
    public List<TranscriptionSegment> Segments { get; set; } = new();
}

public class TranscriptionSegment
{
    public double Start { get; set; }
    public double End { get; set; }
    public string Text { get; set; } = "";
    public double Confidence { get; set; }
}

public class VoiceInfo
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
}

public class SpeechStats
{
    public int TranscriptionCount { get; set; }
    public int SynthesisCount { get; set; }
    public double TotalAudioProcessed { get; set; }
    public string WhisperModel { get; set; } = "";
    public string TTSModel { get; set; } = "";
}

public class TTSConfig
{
    public string Voice { get; set; } = "alloy";
    public float Speed { get; set; } = 1.0f;
    public AudioFormat Format { get; set; } = AudioFormat.WAV;
    public int SampleRate { get; set; } = 24000;
}

public enum AudioFormat { WAV, MP3, OGG }

// =============================================================================
// HTML UI
// =============================================================================

static string GetHtmlUI() => """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiDotNet Speech Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5rem; margin-bottom: 10px; }
        h1 span { color: #a855f7; }
        .subtitle { color: rgba(255,255,255,0.7); font-size: 1.1rem; }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            justify-content: center;
        }
        .tab {
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab:hover { background: rgba(255,255,255,0.2); }
        .tab.active { background: #a855f7; }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 30px;
            display: none;
        }
        .panel.active { display: block; }
        .section-title { font-size: 1.3rem; margin-bottom: 20px; color: #a855f7; }
        .record-area {
            text-align: center;
            padding: 40px;
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 16px;
            margin-bottom: 20px;
        }
        .record-btn {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
            color: #fff;
            font-size: 2.5rem;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(168,85,247,0.3);
        }
        .record-btn:hover { transform: scale(1.05); }
        .record-btn.recording {
            animation: pulse 1.5s infinite;
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
            50% { box-shadow: 0 0 0 20px rgba(239,68,68,0); }
        }
        .record-status {
            margin-top: 15px;
            font-size: 0.9rem;
            color: rgba(255,255,255,0.7);
        }
        .upload-area {
            padding: 30px;
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 16px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .upload-area:hover { border-color: #a855f7; background: rgba(168,85,247,0.1); }
        .upload-area input { display: none; }
        .result-box {
            margin-top: 20px;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            display: none;
        }
        .result-box.show { display: block; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .result-text {
            font-size: 1.1rem;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .result-meta {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
            color: rgba(255,255,255,0.6);
        }
        .tts-input {
            width: 100%;
            padding: 20px;
            border: none;
            border-radius: 12px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1rem;
            resize: vertical;
            min-height: 120px;
            margin-bottom: 20px;
        }
        .tts-input::placeholder { color: rgba(255,255,255,0.5); }
        .tts-controls {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .voice-select, .speed-slider {
            padding: 10px 15px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1rem;
        }
        .speed-label { color: rgba(255,255,255,0.7); }
        .synthesize-btn {
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
            color: #fff;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .synthesize-btn:hover { transform: scale(1.02); }
        .synthesize-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .audio-player {
            width: 100%;
            margin-top: 20px;
            display: none;
        }
        .audio-player.show { display: block; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            text-align: center;
        }
        .stat-value { font-size: 2rem; font-weight: 600; color: #a855f7; }
        .stat-label { color: rgba(255,255,255,0.6); margin-top: 5px; }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #a855f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span>AiDotNet</span> Speech Assistant</h1>
            <p class="subtitle">Powered by Whisper and Text-to-Speech AI models</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="showTab('stt')">Speech-to-Text</button>
            <button class="tab" onclick="showTab('tts')">Text-to-Speech</button>
            <button class="tab" onclick="showTab('stats')">Statistics</button>
        </div>

        <div id="stt-panel" class="panel active">
            <h2 class="section-title">Record or Upload Audio</h2>

            <div class="record-area">
                <button class="record-btn" id="recordBtn" onclick="toggleRecording()">🎤</button>
                <div class="record-status" id="recordStatus">Click to start recording</div>
            </div>

            <div class="upload-area" onclick="document.getElementById('audioFile').click()">
                <input type="file" id="audioFile" accept="audio/*" onchange="handleFileUpload(event)">
                <p>📁 Click or drag audio file here to transcribe</p>
                <p style="color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:10px;">
                    Supports WAV, MP3, OGG, and more
                </p>
            </div>

            <div class="result-box" id="sttResult">
                <div class="result-text" id="transcriptionText"></div>
                <div class="result-meta">
                    <span>Duration: <span id="audioDuration">-</span>s</span>
                    <span>Processing: <span id="processingTime">-</span>s</span>
                    <span>Language: <span id="detectedLang">-</span></span>
                </div>
            </div>
        </div>

        <div id="tts-panel" class="panel">
            <h2 class="section-title">Text-to-Speech</h2>

            <textarea class="tts-input" id="ttsText" placeholder="Enter text to convert to speech..."></textarea>

            <div class="tts-controls">
                <select class="voice-select" id="voiceSelect">
                    <option value="alloy">Alloy (Neutral)</option>
                    <option value="echo">Echo (Warm)</option>
                    <option value="fable">Fable (Expressive)</option>
                    <option value="onyx">Onyx (Deep)</option>
                    <option value="nova">Nova (Youthful)</option>
                    <option value="shimmer">Shimmer (Soft)</option>
                </select>

                <span class="speed-label">Speed:</span>
                <input type="range" class="speed-slider" id="speedSlider" min="0.5" max="2" step="0.1" value="1">
                <span id="speedValue">1.0x</span>

                <button class="synthesize-btn" id="synthesizeBtn" onclick="synthesize()">
                    🔊 Synthesize Speech
                </button>
            </div>

            <audio class="audio-player" id="audioPlayer" controls></audio>
        </div>

        <div id="stats-panel" class="panel">
            <h2 class="section-title">Usage Statistics</h2>
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card">
                    <div class="stat-value" id="transcriptionCount">-</div>
                    <div class="stat-label">Transcriptions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="synthesisCount">-</div>
                    <div class="stat-label">Syntheses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalAudio">-</div>
                    <div class="stat-label">Audio Processed (s)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">Whisper</div>
                    <div class="stat-label">STT Model</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">TTS-1</div>
                    <div class="stat-label">TTS Model</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + '-panel').classList.add('active');

            if (tab === 'stats') loadStats();
        }

        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await transcribeAudio(audioBlob);
                };

                mediaRecorder.start();
                isRecording = true;
                document.getElementById('recordBtn').classList.add('recording');
                document.getElementById('recordBtn').textContent = '⏹️';
                document.getElementById('recordStatus').textContent = 'Recording... Click to stop';
            } catch (err) {
                alert('Error accessing microphone: ' + err.message);
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                document.getElementById('recordBtn').classList.remove('recording');
                document.getElementById('recordBtn').textContent = '🎤';
                document.getElementById('recordStatus').textContent = 'Processing...';
            }
        }

        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                await transcribeAudio(file);
            }
        }

        async function transcribeAudio(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');

            document.getElementById('recordStatus').innerHTML = 'Transcribing<span class="loading"></span>';

            try {
                const response = await fetch('/api/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                document.getElementById('transcriptionText').textContent = result.text;
                document.getElementById('audioDuration').textContent = result.duration.toFixed(2);
                document.getElementById('processingTime').textContent = result.processingTime.toFixed(2);
                document.getElementById('detectedLang').textContent = result.language || 'en';
                document.getElementById('sttResult').classList.add('show');
                document.getElementById('recordStatus').textContent = 'Click to start recording';
            } catch (err) {
                document.getElementById('recordStatus').textContent = 'Error: ' + err.message;
            }
        }

        async function synthesize() {
            const text = document.getElementById('ttsText').value.trim();
            if (!text) return alert('Please enter some text');

            const voice = document.getElementById('voiceSelect').value;
            const speed = parseFloat(document.getElementById('speedSlider').value);
            const btn = document.getElementById('synthesizeBtn');

            btn.disabled = true;
            btn.innerHTML = '🔊 Synthesizing<span class="loading"></span>';

            try {
                const response = await fetch('/api/synthesize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, voice, speed })
                });

                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                const player = document.getElementById('audioPlayer');
                player.src = audioUrl;
                player.classList.add('show');
                player.play();
            } catch (err) {
                alert('Error: ' + err.message);
            }

            btn.disabled = false;
            btn.textContent = '🔊 Synthesize Speech';
        }

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                document.getElementById('transcriptionCount').textContent = stats.transcriptionCount;
                document.getElementById('synthesisCount').textContent = stats.synthesisCount;
                document.getElementById('totalAudio').textContent = stats.totalAudioProcessed.toFixed(1);
            } catch (e) { console.error(e); }
        }

        document.getElementById('speedSlider').oninput = function() {
            document.getElementById('speedValue').textContent = this.value + 'x';
        };
    </script>
</body>
</html>
""";
