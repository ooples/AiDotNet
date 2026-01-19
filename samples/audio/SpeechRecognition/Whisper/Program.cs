using AiDotNet;
using AiDotNet.Audio.Whisper;

Console.WriteLine("=== AiDotNet Whisper Speech Recognition ===");
Console.WriteLine("Automatic Speech Recognition with OpenAI's Whisper\n");

// Whisper model sizes
Console.WriteLine("Available Whisper models:");
Console.WriteLine("  tiny   - 39M params  - Fastest, good accuracy");
Console.WriteLine("  base   - 74M params  - Fast, better accuracy");
Console.WriteLine("  small  - 244M params - Medium speed, good+ accuracy");
Console.WriteLine("  medium - 769M params - Slower, great accuracy");
Console.WriteLine("  large  - 1.5B params - Slowest, best accuracy");
Console.WriteLine();

try
{
    // Load Whisper model
    Console.WriteLine("Loading Whisper model (base)...");

    var whisper = new WhisperModel<float>(
        modelSize: WhisperModelSize.Base,
        language: null,  // Auto-detect language
        useGpu: true);

    Console.WriteLine("  Model loaded successfully\n");

    // Create sample audio data (in real use, load from file)
    var sampleRate = 16000;
    var duration = 5.0;  // seconds
    var audioData = GenerateSampleAudio(sampleRate, duration);

    Console.WriteLine($"Processing audio...");
    Console.WriteLine($"  Duration: {duration:F1} seconds");
    Console.WriteLine($"  Sample rate: {sampleRate} Hz");
    Console.WriteLine($"  Samples: {audioData.Length:N0}\n");

    // Configure transcription options
    var options = new TranscriptionOptions
    {
        Language = null,  // Auto-detect
        Task = TranscriptionTask.Transcribe,
        ReturnTimestamps = true,
        WordTimestamps = true,
        Temperature = 0.0f,  // Greedy decoding
        BestOf = 1
    };

    Console.WriteLine("Transcribing...\n");

    // Transcribe audio
    var result = await whisper.TranscribeAsync(audioData, sampleRate, options);

    // Display results
    if (result.DetectedLanguage != null)
    {
        Console.WriteLine($"Language detected: {result.DetectedLanguage} (confidence: {result.LanguageConfidence:F2})\n");
    }

    Console.WriteLine("Transcription:");
    Console.WriteLine("─────────────────────────────────────");
    Console.WriteLine(result.Text);
    Console.WriteLine();

    // Display timestamps if available
    if (result.Segments != null && result.Segments.Any())
    {
        Console.WriteLine("Segments with timestamps:");
        Console.WriteLine("─────────────────────────────────────");
        foreach (var segment in result.Segments.Take(10))
        {
            Console.WriteLine($"  [{segment.Start:F2}s - {segment.End:F2}s] {segment.Text}");
        }

        if (result.Segments.Count() > 10)
            Console.WriteLine($"  ... and {result.Segments.Count() - 10} more segments");
    }

    // Word-level timestamps
    if (result.Words != null && result.Words.Any())
    {
        Console.WriteLine("\nWord-level timestamps:");
        Console.WriteLine("─────────────────────────────────────");
        foreach (var word in result.Words.Take(10))
        {
            Console.WriteLine($"  [{word.Start:F2}s - {word.End:F2}s] {word.Text}");
        }

        if (result.Words.Count() > 10)
            Console.WriteLine($"  ... and {result.Words.Count() - 10} more words");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full Whisper implementation requires model weights.");
    Console.WriteLine($"This sample demonstrates the API pattern for speech recognition.");
    Console.WriteLine($"\nError details: {ex.Message}");

    // Show what the API would look like
    DemoApiUsage();
}

Console.WriteLine("\n=== Sample Complete ===");

// Generate sample audio (sine wave with speech-like amplitude modulation)
static float[] GenerateSampleAudio(int sampleRate, double duration)
{
    var samples = (int)(sampleRate * duration);
    var audio = new float[samples];
    var random = new Random(42);

    for (int i = 0; i < samples; i++)
    {
        double t = (double)i / sampleRate;

        // Mix of frequencies to simulate speech
        double signal = 0;
        signal += 0.3 * Math.Sin(2 * Math.PI * 200 * t);   // Fundamental
        signal += 0.2 * Math.Sin(2 * Math.PI * 400 * t);   // Harmonic 1
        signal += 0.1 * Math.Sin(2 * Math.PI * 600 * t);   // Harmonic 2

        // Amplitude modulation (speech envelope)
        double envelope = 0.5 + 0.5 * Math.Sin(2 * Math.PI * 3 * t);

        // Add some noise
        double noise = (random.NextDouble() - 0.5) * 0.05;

        audio[i] = (float)(signal * envelope + noise);
    }

    return audio;
}

// Demo API usage
static void DemoApiUsage()
{
    Console.WriteLine("\nAPI Usage Demo:");
    Console.WriteLine("─────────────────────────────────────");
    Console.WriteLine(@"
// Load model
var whisper = new WhisperModel<float>(WhisperModelSize.Base);

// Transcribe from file
var result = await whisper.TranscribeFileAsync(""audio.wav"");
Console.WriteLine(result.Text);

// Transcribe from audio array
var audioData = LoadAudio(""speech.mp3"");
var result = await whisper.TranscribeAsync(audioData, sampleRate: 16000);

// With options
var result = await whisper.TranscribeAsync(audioData, 16000, new TranscriptionOptions
{
    Language = ""en"",
    ReturnTimestamps = true,
    WordTimestamps = true,
    Task = TranscriptionTask.Translate  // Translate to English
});

// Streaming transcription
await foreach (var segment in whisper.TranscribeStreamAsync(audioStream))
{
    Console.Write(segment.Text);
}
");
}

// Supporting classes (simplified for sample)
public enum WhisperModelSize
{
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3
}

public enum TranscriptionTask
{
    Transcribe,
    Translate
}

public class TranscriptionOptions
{
    public string? Language { get; set; }
    public TranscriptionTask Task { get; set; } = TranscriptionTask.Transcribe;
    public bool ReturnTimestamps { get; set; }
    public bool WordTimestamps { get; set; }
    public float Temperature { get; set; } = 0.0f;
    public int BestOf { get; set; } = 1;
}

public class TranscriptionResult
{
    public string Text { get; set; } = "";
    public string? DetectedLanguage { get; set; }
    public float LanguageConfidence { get; set; }
    public IEnumerable<Segment>? Segments { get; set; }
    public IEnumerable<Word>? Words { get; set; }
}

public class Segment
{
    public float Start { get; set; }
    public float End { get; set; }
    public string Text { get; set; } = "";
}

public class Word
{
    public float Start { get; set; }
    public float End { get; set; }
    public string Text { get; set; } = "";
    public float Confidence { get; set; }
}

// Simplified Whisper model for sample
public class WhisperModel<T>
{
    private readonly WhisperModelSize _modelSize;
    private readonly string? _language;
    private readonly bool _useGpu;

    public WhisperModel(WhisperModelSize modelSize, string? language = null, bool useGpu = true)
    {
        _modelSize = modelSize;
        _language = language;
        _useGpu = useGpu;
    }

    public Task<TranscriptionResult> TranscribeAsync(float[] audio, int sampleRate, TranscriptionOptions? options = null)
    {
        // In real implementation, this would:
        // 1. Preprocess audio (resample, normalize)
        // 2. Extract mel spectrogram features
        // 3. Run encoder
        // 4. Decode with attention
        // 5. Post-process output

        // For demo, return a simulated result
        return Task.FromResult(new TranscriptionResult
        {
            Text = "This is a sample transcription from the Whisper model. " +
                   "In a real implementation, this would contain the actual transcribed text from the audio.",
            DetectedLanguage = "English",
            LanguageConfidence = 0.98f,
            Segments = new[]
            {
                new Segment { Start = 0.0f, End = 2.5f, Text = "This is a sample transcription" },
                new Segment { Start = 2.5f, End = 4.0f, Text = "from the Whisper model." }
            },
            Words = new[]
            {
                new Word { Start = 0.0f, End = 0.3f, Text = "This", Confidence = 0.99f },
                new Word { Start = 0.3f, End = 0.5f, Text = "is", Confidence = 0.98f },
                new Word { Start = 0.5f, End = 0.7f, Text = "a", Confidence = 0.99f },
                new Word { Start = 0.7f, End = 1.2f, Text = "sample", Confidence = 0.97f },
                new Word { Start = 1.2f, End = 2.0f, Text = "transcription", Confidence = 0.95f }
            }
        });
    }
}
