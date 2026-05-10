using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration for the TIMIT acoustic-phonetic continuous-speech corpus loader (Garofolo et al. 1993).
/// </summary>
/// <remarks>
/// <para>
/// TIMIT — 6,300 sentences from 630 speakers (8 dialect regions × ~80 speakers).
/// The classic phoneme-recognition benchmark; widely used in early speech-recognition
/// research. Contains sphere-format WAV + word/phoneme alignment files.
/// </para>
/// <para>
/// <b>Commercial license required</b> — TIMIT is distributed by LDC (catalog
/// LDC93S1) and requires a paid LDC membership. <see cref="AutoDownload"/>
/// is unavailable; this loader expects the user to manually extract the
/// distribution.
/// </para>
/// </remarks>
public sealed class TimitDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    /// <summary>Always false — TIMIT requires LDC membership.</summary>
    public bool AutoDownload { get; set; } = false;
    public int SampleRate { get; set; } = 16000;
    public int MaxAudioSamples { get; set; } = 16000 * 8;
    public int MaxTextLength { get; set; } = 256;
    public int VocabularySize { get; set; } = 1024;
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "SampleRate must be positive.");
        if (MaxAudioSamples <= 0) throw new ArgumentOutOfRangeException(nameof(MaxAudioSamples), "MaxAudioSamples must be positive.");
        if (MaxTextLength <= 0) throw new ArgumentOutOfRangeException(nameof(MaxTextLength), "MaxTextLength must be positive.");
        if (VocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(VocabularySize), "VocabularySize must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
