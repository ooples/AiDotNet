using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the VoxPopuli data loader.
/// </summary>
/// <remarks>
/// <para>
/// VoxPopuli is a large-scale multilingual speech corpus from European Parliament recordings.
/// Contains 400K+ hours of unlabeled speech and 1800+ hours of transcribed speech in 23 languages.
/// </para>
/// </remarks>
public sealed class VoxPopuliDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 16000.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Maximum audio duration in seconds. Default is 20.</summary>
    public double MaxDurationSeconds { get; set; } = 20.0;
    /// <summary>Language code. Default is "en".</summary>
    public string Language { get; set; } = "en";

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "SampleRate must be positive.");
        if (MaxDurationSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(MaxDurationSeconds), "MaxDurationSeconds must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
