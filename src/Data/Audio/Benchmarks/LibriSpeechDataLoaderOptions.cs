using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the LibriSpeech data loader.
/// </summary>
/// <remarks>
/// <para>
/// LibriSpeech is a corpus of ~1000 hours of 16kHz English speech derived from audiobooks.
/// Subsets: train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other.
/// Audio is stored as FLAC files with corresponding text transcriptions.
/// </para>
/// </remarks>
public sealed class LibriSpeechDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;

    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Sample rate in Hz. Default is 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Maximum audio duration in seconds. Clips are padded/truncated. Default is 10.</summary>
    public double MaxDurationSeconds { get; set; } = 10.0;

    /// <summary>
    /// LibriSpeech subset. Default is "train-clean-100".
    /// Options: "train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other".
    /// </summary>
    public string Subset { get; set; } = "train-clean-100";

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "SampleRate must be positive.");
        if (MaxDurationSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(MaxDurationSeconds), "MaxDurationSeconds must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
