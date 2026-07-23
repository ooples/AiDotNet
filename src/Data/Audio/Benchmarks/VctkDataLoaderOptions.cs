using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration for the VCTK Corpus 0.92 multi-speaker TTS loader (Yamagishi et al. 2019).
/// </summary>
/// <remarks>
/// <para>
/// VCTK — 110 English speakers × ~400 utterances each (~44 hours) at 48 kHz.
/// Standard multi-speaker TTS / voice-cloning corpus.
/// </para>
/// </remarks>
public sealed class VctkDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    /// <summary>Comma-separated speaker-ID filter (e.g. "p225,p226"). Null = all speakers.</summary>
    public string? SpeakerFilter { get; set; }
    public int MaxAudioSamples { get; set; } = 48_000 * 8;   // 8 sec at 48 kHz
    public int MaxTextLength { get; set; } = 256;
    public int VocabularySize { get; set; } = 256;
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
