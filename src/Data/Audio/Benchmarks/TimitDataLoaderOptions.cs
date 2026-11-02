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
    /// <summary>Gets or sets which split of the dataset is loaded.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Gets or sets the dataset root directory; null uses the default data directory for this dataset.</summary>
    public string? DataPath { get; set; }
    /// <summary>Always false — TIMIT requires LDC membership.</summary>
    public bool AutoDownload { get; set; } = false;
    /// <summary>Gets or sets the sample rate TIMIT is recorded at (16 kHz). The loader reads the WAV samples as stored and does not resample, so it is not read.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the waveform length, in samples, each clip is zero-padded or truncated to.</summary>
    public int MaxAudioSamples { get; set; } = 16000 * 8;
    /// <summary>Gets or sets the token length each transcript is padded or truncated to.</summary>
    public int MaxTextLength { get; set; } = 256;
    /// <summary>Gets or sets the most distinct tokens in the vocabulary built from the loaded text.</summary>
    public int VocabularySize { get; set; } = 1024;
    /// <summary>Gets or sets the most samples loaded; null loads the whole split.</summary>
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
