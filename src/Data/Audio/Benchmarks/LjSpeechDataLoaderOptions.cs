using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the LJSpeech 1.1 data loader (Ito &amp; Johnson 2017).
/// </summary>
/// <remarks>
/// <para>
/// LJSpeech is the canonical single-speaker English TTS corpus — 13,100
/// audiobook clips (≈ 24 hours) from a single female narrator at 22,050 Hz.
/// Used as the default TTS training corpus by Tacotron, FastSpeech, VITS,
/// and most subsequent neural TTS papers.
/// </para>
/// </remarks>
public sealed class LjSpeechDataLoaderOptions
{
    /// <summary>Dataset split. LJSpeech has no canonical splits; this loader uses a deterministic 90/5/5 by row index.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum waveform length in samples (zero-padded). Default 22050 * 10 = 10s.</summary>
    public int MaxAudioSamples { get; set; } = 22_050 * 10;
    /// <summary>Maximum transcript length in tokens (input text). Default 256.</summary>
    public int MaxTextLength { get; set; } = 256;
    /// <summary>Vocabulary size for the text tokenizer. Default 256 (character-level effectively).</summary>
    public int VocabularySize { get; set; } = 256;
    /// <summary>Use the normalized text (numbers/dates expanded). Default true (recommended for TTS).</summary>
    public bool UseNormalizedText { get; set; } = true;
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MaxAudioSamples <= 0) throw new ArgumentOutOfRangeException(nameof(MaxAudioSamples), "MaxAudioSamples must be positive.");
        if (MaxTextLength <= 0) throw new ArgumentOutOfRangeException(nameof(MaxTextLength), "MaxTextLength must be positive.");
        if (VocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(VocabularySize), "VocabularySize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
