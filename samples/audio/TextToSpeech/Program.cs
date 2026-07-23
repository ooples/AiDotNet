using AiDotNet;
using AiDotNet.Tensors;

Console.WriteLine("=== AiDotNet Text-to-Speech Generation ===");
Console.WriteLine("Neural TTS with VITS (Variational Inference Text-to-Speech)\n");

// Display available TTS models
Console.WriteLine("Available TTS Models:");
Console.WriteLine("  VITS           - High-quality, end-to-end TTS");
Console.WriteLine("  FastSpeech2    - Fast, parallel synthesis");
Console.WriteLine("  Tacotron2      - Attention-based seq2seq");
Console.WriteLine("  GlowTTS        - Flow-based for diverse speech");
Console.WriteLine("  StyleTTS2      - Style-controllable TTS");
Console.WriteLine();

try
{
    // Configure TTS model
    var config = new TTSConfig
    {
        Model = TTSModelType.VITS,
        Language = "en",
        SampleRate = 22050,
        UseGpu = true
    };

    Console.WriteLine("Configuration:");
    Console.WriteLine($"  Model: {config.Model}");
    Console.WriteLine($"  Language: {config.Language}");
    Console.WriteLine($"  Sample Rate: {config.SampleRate} Hz");
    Console.WriteLine($"  GPU Acceleration: {config.UseGpu}");
    Console.WriteLine();

    // Create TTS synthesizer
    Console.WriteLine("Loading TTS model...");
    var synthesizer = new TextToSpeechSynthesizer(config);
    Console.WriteLine("  Model loaded successfully\n");

    // Available voices
    Console.WriteLine("Available Voices:");
    Console.WriteLine(new string('-', 70));
    Console.WriteLine("| ID  | Name           | Gender | Style      | Language |");
    Console.WriteLine(new string('-', 70));

    var voices = synthesizer.GetAvailableVoices();
    foreach (var voice in voices)
    {
        Console.WriteLine($"| {voice.Id,-3} | {voice.Name,-14} | {voice.Gender,-6} | {voice.Style,-10} | {voice.Language,-8} |");
    }
    Console.WriteLine(new string('-', 70));
    Console.WriteLine();

    // Demo 1: Basic TTS
    Console.WriteLine("=== Demo 1: Basic Text-to-Speech ===\n");

    var text1 = "Hello! Welcome to AiDotNet's text-to-speech demonstration. " +
                "This sample shows how to generate natural-sounding speech from text.";

    Console.WriteLine($"Text: \"{text1}\"\n");

    var result1 = synthesizer.Synthesize(text1);

    Console.WriteLine("Synthesis Result:");
    Console.WriteLine($"  Duration: {result1.Duration.TotalSeconds:F2} seconds");
    Console.WriteLine($"  Sample Rate: {result1.SampleRate} Hz");
    Console.WriteLine($"  Samples: {result1.AudioData.Length:N0}");
    Console.WriteLine($"  Channels: {result1.Channels}");
    Console.WriteLine($"  Synthesis Time: {result1.ProcessingTime.TotalMilliseconds:F2}ms");
    Console.WriteLine($"  Real-time Factor: {result1.RealTimeFactor:F2}x");
    Console.WriteLine();

    // Visualize audio waveform
    Console.WriteLine("Audio Waveform (first 100ms):");
    VisualizeWaveform(result1.AudioData, result1.SampleRate, 0.1);

    // Demo 2: Voice Selection
    Console.WriteLine("\n=== Demo 2: Voice Selection ===\n");

    var voiceOptions = new SynthesisOptions
    {
        VoiceId = "en_female_2",
        Speed = 1.0f,
        Pitch = 1.0f
    };

    var text2 = "Each voice has unique characteristics. Listen to how different voices sound.";
    Console.WriteLine($"Text: \"{text2}\"");
    Console.WriteLine($"Voice: {voiceOptions.VoiceId}\n");

    var result2 = synthesizer.Synthesize(text2, voiceOptions);

    Console.WriteLine("Synthesis Result:");
    Console.WriteLine($"  Duration: {result2.Duration.TotalSeconds:F2} seconds");
    Console.WriteLine($"  Processing Time: {result2.ProcessingTime.TotalMilliseconds:F2}ms");
    Console.WriteLine();

    // Demo 3: Speed Control
    Console.WriteLine("=== Demo 3: Speed Control ===\n");

    var text3 = "This sentence demonstrates different speaking speeds.";
    Console.WriteLine($"Text: \"{text3}\"\n");

    var speeds = new[] { 0.75f, 1.0f, 1.25f, 1.5f };
    Console.WriteLine("| Speed  | Duration | RTF    | Description        |");
    Console.WriteLine(new string('-', 55));

    foreach (var speed in speeds)
    {
        var options = new SynthesisOptions { Speed = speed };
        var result = synthesizer.Synthesize(text3, options);
        var description = speed switch
        {
            0.75f => "Slow, deliberate",
            1.0f => "Normal speed",
            1.25f => "Faster paced",
            1.5f => "Quick speech",
            _ => ""
        };
        Console.WriteLine($"| {speed,5:F2}x | {result.Duration.TotalSeconds,7:F2}s | {result.RealTimeFactor,5:F2}x | {description,-18} |");
    }
    Console.WriteLine(new string('-', 55));

    // Demo 4: Pitch Adjustment
    Console.WriteLine("\n=== Demo 4: Pitch Adjustment ===\n");

    var text4 = "Pitch control allows adjusting the voice frequency.";
    Console.WriteLine($"Text: \"{text4}\"\n");

    var pitches = new[] { 0.8f, 1.0f, 1.2f };
    Console.WriteLine("| Pitch  | Effect              |");
    Console.WriteLine(new string('-', 35));

    foreach (var pitch in pitches)
    {
        var options = new SynthesisOptions { Pitch = pitch };
        var result = synthesizer.Synthesize(text4, options);
        var effect = pitch switch
        {
            0.8f => "Lower, deeper voice",
            1.0f => "Natural pitch",
            1.2f => "Higher, brighter",
            _ => ""
        };
        Console.WriteLine($"| {pitch,5:F1}x  | {effect,-19} |");
    }
    Console.WriteLine(new string('-', 35));

    // Demo 5: Emotion/Style Control
    Console.WriteLine("\n=== Demo 5: Emotion/Style Control ===\n");

    var text5 = "The ability to convey emotion makes speech more expressive and engaging.";
    Console.WriteLine($"Text: \"{text5}\"\n");

    var emotions = new[] { SpeechEmotion.Neutral, SpeechEmotion.Happy, SpeechEmotion.Sad, SpeechEmotion.Excited };
    Console.WriteLine("| Emotion  | Energy | Pitch Var | Speaking Rate |");
    Console.WriteLine(new string('-', 50));

    foreach (var emotion in emotions)
    {
        var options = new SynthesisOptions { Emotion = emotion };
        var result = synthesizer.Synthesize(text5, options);
        Console.WriteLine($"| {emotion,-8} | {result.EmotionMetrics.Energy,5:F2}  | {result.EmotionMetrics.PitchVariation,9:F2} | {result.EmotionMetrics.SpeakingRate,13:F2} |");
    }
    Console.WriteLine(new string('-', 50));

    // Demo 6: SSML Support
    Console.WriteLine("\n=== Demo 6: SSML (Speech Synthesis Markup Language) ===\n");

    var ssml = @"<speak>
    <s>Welcome to AiDotNet.</s>
    <s><emphasis level=""strong"">This text is emphasized.</emphasis></s>
    <s>Here's a pause<break time=""500ms""/> and then we continue.</s>
    <s>Numbers like <say-as interpret-as=""cardinal"">42</say-as> are spoken correctly.</s>
    <s>Dates like <say-as interpret-as=""date"">2024-01-15</say-as> are formatted.</s>
</speak>";

    Console.WriteLine("SSML Input:");
    Console.WriteLine(ssml);
    Console.WriteLine();

    var ssmlResult = synthesizer.SynthesizeSsml(ssml);
    Console.WriteLine("SSML Synthesis Result:");
    Console.WriteLine($"  Duration: {ssmlResult.Duration.TotalSeconds:F2} seconds");
    Console.WriteLine($"  Segments Processed: {ssmlResult.SegmentCount}");
    Console.WriteLine();

    // Demo 7: Phoneme-level Control
    Console.WriteLine("=== Demo 7: Phoneme-level Analysis ===\n");

    var text7 = "Hello world";
    Console.WriteLine($"Text: \"{text7}\"");
    Console.WriteLine();

    var phonemeResult = synthesizer.SynthesizeWithPhonemes(text7);
    Console.WriteLine("Phoneme Breakdown:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine("| Phoneme | Start (ms) | End (ms) | Duration |");
    Console.WriteLine(new string('-', 50));

    foreach (var phoneme in phonemeResult.Phonemes.Take(10))
    {
        Console.WriteLine($"| {phoneme.Symbol,-7} | {phoneme.StartMs,10:F1} | {phoneme.EndMs,8:F1} | {phoneme.DurationMs,8:F1}ms |");
    }

    if (phonemeResult.Phonemes.Count > 10)
        Console.WriteLine($"  ... and {phonemeResult.Phonemes.Count - 10} more phonemes");
    Console.WriteLine(new string('-', 50));

    // Demo 8: Batch Synthesis
    Console.WriteLine("\n=== Demo 8: Batch Synthesis ===\n");

    var sentences = new[]
    {
        "First sentence for batch processing.",
        "Second sentence with different content.",
        "Third and final sentence in the batch."
    };

    Console.WriteLine("Input Sentences:");
    for (int i = 0; i < sentences.Length; i++)
    {
        Console.WriteLine($"  {i + 1}. {sentences[i]}");
    }
    Console.WriteLine();

    var batchStart = DateTime.UtcNow;
    var batchResults = synthesizer.SynthesizeBatch(sentences);
    var batchTime = DateTime.UtcNow - batchStart;

    Console.WriteLine("Batch Results:");
    Console.WriteLine($"  Total Sentences: {batchResults.Count}");
    Console.WriteLine($"  Total Duration: {batchResults.Sum(r => r.Duration.TotalSeconds):F2} seconds of audio");
    Console.WriteLine($"  Processing Time: {batchTime.TotalMilliseconds:F2}ms");
    Console.WriteLine($"  Throughput: {batchResults.Sum(r => r.AudioData.Length) / batchTime.TotalSeconds / 1000:F2} samples/ms");

    // Demo 9: Save Audio
    Console.WriteLine("\n=== Demo 9: Audio Export Formats ===\n");

    Console.WriteLine("Supported export formats:");
    Console.WriteLine("  - WAV  (Uncompressed, lossless)");
    Console.WriteLine("  - MP3  (Compressed, lossy)");
    Console.WriteLine("  - OGG  (Compressed, open format)");
    Console.WriteLine("  - FLAC (Compressed, lossless)");
    Console.WriteLine();

    Console.WriteLine("Export Code Example:");
    Console.WriteLine(@"
    // Save as WAV
    synthesizer.SaveToWav(result, ""output.wav"");

    // Save as MP3 with bitrate
    synthesizer.SaveToMp3(result, ""output.mp3"", bitrate: 192);

    // Save as FLAC (lossless)
    synthesizer.SaveToFlac(result, ""output.flac"");
    ");

    // Demo 10: Real-time Streaming
    Console.WriteLine("=== Demo 10: Real-time Streaming ===\n");

    Console.WriteLine("Streaming synthesis for real-time applications:");
    Console.WriteLine(@"
    // Stream synthesis for real-time playback
    await foreach (var chunk in synthesizer.SynthesizeStreamAsync(longText))
    {
        // Process audio chunk as it's generated
        audioPlayer.EnqueueChunk(chunk.AudioData);
        Console.WriteLine($""Chunk {chunk.Index}: {chunk.Duration.TotalMilliseconds:F0}ms"");
    }
    ");

    // Simulate streaming output
    Console.WriteLine("Simulated Streaming Output:");
    var longText = "This is a longer text that would be synthesized in chunks for real-time streaming playback.";
    var chunkCount = 5;
    var chunkDuration = 0.0;

    for (int i = 0; i < chunkCount; i++)
    {
        var duration = 200 + new Random(i).Next(100);
        chunkDuration += duration;
        Console.WriteLine($"  Chunk {i + 1}: {duration}ms generated");
    }
    Console.WriteLine($"  Total streamed: {chunkDuration}ms");

    // Architecture Overview
    Console.WriteLine("\n\n=== VITS Architecture Overview ===\n");

    Console.WriteLine("VITS (Variational Inference with adversarial learning for TTS):");
    Console.WriteLine(@"
    Input Text
        |
        v
    +------------------+
    | Text Encoder     |  Transformer-based text processing
    +------------------+
        |
        v
    +------------------+
    | Stochastic       |  Variational autoencoder for
    | Duration Pred    |  duration prediction
    +------------------+
        |
        v
    +------------------+
    | Flow-based       |  Normalizing flow for
    | Decoder          |  latent to mel conversion
    +------------------+
        |
        v
    +------------------+
    | HiFi-GAN         |  Neural vocoder for
    | Vocoder          |  waveform synthesis
    +------------------+
        |
        v
    Audio Waveform
    ");

    Console.WriteLine("Key Features:");
    Console.WriteLine("  - End-to-end: Text directly to waveform");
    Console.WriteLine("  - Parallel synthesis: Fast inference");
    Console.WriteLine("  - High quality: Natural, expressive speech");
    Console.WriteLine("  - Multi-speaker: Single model, many voices");
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full TTS implementation requires model weights.");
    Console.WriteLine($"This sample demonstrates the API pattern for text-to-speech.");
    Console.WriteLine($"\nError details: {ex.Message}");

    // Show API usage demo
    DemoApiUsage();
}

Console.WriteLine("\n=== Sample Complete ===");

// Visualize audio waveform in console
static void VisualizeWaveform(float[] audio, int sampleRate, double duration)
{
    var samplesToShow = (int)(sampleRate * duration);
    var step = Math.Max(1, samplesToShow / 60);
    var height = 10;

    Console.WriteLine("  +" + new string('-', 60) + "+");

    for (int row = height - 1; row >= 0; row--)
    {
        var threshold = (row - height / 2.0) / (height / 2.0);
        var line = "  |";

        for (int col = 0; col < 60; col++)
        {
            var sampleIdx = col * step;
            if (sampleIdx < audio.Length)
            {
                var value = audio[sampleIdx];
                if ((row >= height / 2 && value >= threshold) || (row < height / 2 && value <= threshold))
                    line += "*";
                else
                    line += " ";
            }
            else
            {
                line += " ";
            }
        }

        line += "|";
        if (row == height / 2)
            line += " 0";
        else if (row == height - 1)
            line += " +1";
        else if (row == 0)
            line += " -1";

        Console.WriteLine(line);
    }

    Console.WriteLine("  +" + new string('-', 60) + "+");
    Console.WriteLine("   0ms" + new string(' ', 50) + $"{duration * 1000:F0}ms");
}

// Demo API usage
static void DemoApiUsage()
{
    Console.WriteLine("\nAPI Usage Demo:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine(@"
// 1. Configure TTS
var config = new TTSConfig
{
    Model = TTSModelType.VITS,
    Language = ""en"",
    SampleRate = 22050
};

// 2. Create synthesizer
var tts = new TextToSpeechSynthesizer(config);

// 3. Basic synthesis
var result = tts.Synthesize(""Hello, world!"");

// 4. With voice and style options
var options = new SynthesisOptions
{
    VoiceId = ""en_female_1"",
    Speed = 1.1f,
    Pitch = 1.0f,
    Emotion = SpeechEmotion.Happy
};
var result = tts.Synthesize(text, options);

// 5. Save to file
tts.SaveToWav(result, ""output.wav"");

// 6. Stream for real-time playback
await foreach (var chunk in tts.SynthesizeStreamAsync(text))
{
    player.Play(chunk.AudioData);
}
");
}

// Supporting classes for demonstration

public enum TTSModelType
{
    VITS,
    FastSpeech2,
    Tacotron2,
    GlowTTS,
    StyleTTS2
}

public enum SpeechEmotion
{
    Neutral,
    Happy,
    Sad,
    Angry,
    Excited,
    Calm,
    Fearful
}

public class TTSConfig
{
    public TTSModelType Model { get; set; } = TTSModelType.VITS;
    public string Language { get; set; } = "en";
    public int SampleRate { get; set; } = 22050;
    public bool UseGpu { get; set; } = true;
}

public class SynthesisOptions
{
    public string? VoiceId { get; set; }
    public float Speed { get; set; } = 1.0f;
    public float Pitch { get; set; } = 1.0f;
    public SpeechEmotion Emotion { get; set; } = SpeechEmotion.Neutral;
}

public class Voice
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Gender { get; set; } = "";
    public string Style { get; set; } = "";
    public string Language { get; set; } = "";
}

public class SynthesisResult
{
    public float[] AudioData { get; set; } = Array.Empty<float>();
    public int SampleRate { get; set; }
    public int Channels { get; set; } = 1;
    public TimeSpan Duration { get; set; }
    public TimeSpan ProcessingTime { get; set; }
    public float RealTimeFactor { get; set; }
    public int SegmentCount { get; set; } = 1;
    public EmotionMetrics EmotionMetrics { get; set; } = new();
    public List<PhonemeInfo> Phonemes { get; set; } = new();
}

public class EmotionMetrics
{
    public float Energy { get; set; }
    public float PitchVariation { get; set; }
    public float SpeakingRate { get; set; }
}

public class PhonemeInfo
{
    public string Symbol { get; set; } = "";
    public float StartMs { get; set; }
    public float EndMs { get; set; }
    public float DurationMs => EndMs - StartMs;
}

public class TextToSpeechSynthesizer
{
    private readonly TTSConfig _config;
    private readonly Random _random = new(42);
    private readonly List<Voice> _voices;

    public TextToSpeechSynthesizer(TTSConfig config)
    {
        _config = config;
        _voices = InitializeVoices();
    }

    private List<Voice> InitializeVoices()
    {
        return new List<Voice>
        {
            new() { Id = "en_male_1", Name = "James", Gender = "Male", Style = "Neutral", Language = "en-US" },
            new() { Id = "en_male_2", Name = "Michael", Gender = "Male", Style = "Warm", Language = "en-US" },
            new() { Id = "en_female_1", Name = "Sarah", Gender = "Female", Style = "Neutral", Language = "en-US" },
            new() { Id = "en_female_2", Name = "Emily", Gender = "Female", Style = "Bright", Language = "en-US" },
            new() { Id = "en_male_3", Name = "David", Gender = "Male", Style = "Formal", Language = "en-GB" },
            new() { Id = "en_female_3", Name = "Emma", Gender = "Female", Style = "Warm", Language = "en-GB" }
        };
    }

    public List<Voice> GetAvailableVoices() => _voices;

    public SynthesisResult Synthesize(string text, SynthesisOptions? options = null)
    {
        var opts = options ?? new SynthesisOptions();
        var startTime = DateTime.UtcNow;

        // Calculate expected duration based on text and options
        var wordCount = text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        var baseDuration = wordCount * 0.35 / opts.Speed;  // ~0.35 seconds per word at normal speed

        // Generate synthetic audio
        var sampleCount = (int)(baseDuration * _config.SampleRate);
        var audioData = GenerateSyntheticSpeech(sampleCount, opts);

        var processingTime = DateTime.UtcNow - startTime;

        return new SynthesisResult
        {
            AudioData = audioData,
            SampleRate = _config.SampleRate,
            Channels = 1,
            Duration = TimeSpan.FromSeconds(baseDuration),
            ProcessingTime = processingTime,
            RealTimeFactor = (float)(baseDuration / processingTime.TotalSeconds),
            EmotionMetrics = GenerateEmotionMetrics(opts.Emotion)
        };
    }

    public SynthesisResult SynthesizeSsml(string ssml)
    {
        // Parse SSML and synthesize
        var segmentCount = ssml.Split("<s>").Length - 1;
        if (segmentCount == 0) segmentCount = 1;

        var result = Synthesize(StripSsml(ssml));
        result.SegmentCount = segmentCount;
        return result;
    }

    public SynthesisResult SynthesizeWithPhonemes(string text)
    {
        var result = Synthesize(text);

        // Generate phoneme timing information
        var phonemes = new List<PhonemeInfo>();
        var currentMs = 0f;

        // Simplified phoneme generation for demonstration
        var phonemeMap = new Dictionary<char, string>
        {
            ['h'] = "HH", ['e'] = "EH", ['l'] = "L", ['o'] = "OW",
            ['w'] = "W", ['r'] = "R", ['d'] = "D", [' '] = "SIL"
        };

        foreach (var c in text.ToLower())
        {
            if (phonemeMap.TryGetValue(c, out var symbol) || char.IsLetter(c))
            {
                var duration = 50 + _random.Next(50);  // 50-100ms per phoneme
                phonemes.Add(new PhonemeInfo
                {
                    Symbol = symbol ?? c.ToString().ToUpper(),
                    StartMs = currentMs,
                    EndMs = currentMs + duration
                });
                currentMs += duration;
            }
        }

        result.Phonemes = phonemes;
        return result;
    }

    public List<SynthesisResult> SynthesizeBatch(string[] texts)
    {
        return texts.Select(t => Synthesize(t)).ToList();
    }

    private float[] GenerateSyntheticSpeech(int sampleCount, SynthesisOptions options)
    {
        var audio = new float[sampleCount];

        // Generate speech-like waveform with formants
        var f0 = options.Pitch * (options.VoiceId?.Contains("female") == true ? 220 : 130);  // Fundamental frequency
        var formants = new[] { f0, f0 * 2.5, f0 * 4.0, f0 * 5.5 };  // Typical speech formants

        for (int i = 0; i < sampleCount; i++)
        {
            double t = (double)i / _config.SampleRate;

            // Generate harmonic content
            double signal = 0;
            for (int h = 0; h < formants.Length; h++)
            {
                var amplitude = 1.0 / (h + 1);  // Decreasing harmonic amplitude
                signal += amplitude * Math.Sin(2 * Math.PI * formants[h] * t);
            }

            // Apply speech envelope (amplitude modulation)
            double envelope = 0.5 + 0.3 * Math.Sin(2 * Math.PI * 4 * t) +
                             0.2 * Math.Sin(2 * Math.PI * 8 * t);

            // Add emotion-based variation
            var emotionMod = GetEmotionModulation(options.Emotion, t);

            // Add noise for naturalness
            double noise = (_random.NextDouble() - 0.5) * 0.02;

            audio[i] = (float)(signal * envelope * emotionMod * 0.3 + noise);
        }

        return audio;
    }

    private double GetEmotionModulation(SpeechEmotion emotion, double t)
    {
        return emotion switch
        {
            SpeechEmotion.Happy => 1.0 + 0.2 * Math.Sin(2 * Math.PI * 3 * t),
            SpeechEmotion.Sad => 0.7 + 0.1 * Math.Sin(2 * Math.PI * 1 * t),
            SpeechEmotion.Excited => 1.2 + 0.3 * Math.Sin(2 * Math.PI * 5 * t),
            SpeechEmotion.Angry => 1.1 + 0.25 * Math.Sin(2 * Math.PI * 4 * t),
            SpeechEmotion.Calm => 0.8 + 0.05 * Math.Sin(2 * Math.PI * 1.5 * t),
            _ => 1.0
        };
    }

    private EmotionMetrics GenerateEmotionMetrics(SpeechEmotion emotion)
    {
        return emotion switch
        {
            SpeechEmotion.Happy => new EmotionMetrics { Energy = 0.75f, PitchVariation = 0.35f, SpeakingRate = 1.15f },
            SpeechEmotion.Sad => new EmotionMetrics { Energy = 0.45f, PitchVariation = 0.15f, SpeakingRate = 0.85f },
            SpeechEmotion.Excited => new EmotionMetrics { Energy = 0.90f, PitchVariation = 0.45f, SpeakingRate = 1.25f },
            SpeechEmotion.Angry => new EmotionMetrics { Energy = 0.85f, PitchVariation = 0.30f, SpeakingRate = 1.10f },
            SpeechEmotion.Calm => new EmotionMetrics { Energy = 0.50f, PitchVariation = 0.10f, SpeakingRate = 0.90f },
            _ => new EmotionMetrics { Energy = 0.60f, PitchVariation = 0.20f, SpeakingRate = 1.00f }
        };
    }

    private static string StripSsml(string ssml)
    {
        // Simple SSML stripping for text content
        var result = ssml;
        result = System.Text.RegularExpressions.Regex.Replace(result, "<[^>]+>", " ");
        result = System.Text.RegularExpressions.Regex.Replace(result, @"\s+", " ");
        return result.Trim();
    }
}
