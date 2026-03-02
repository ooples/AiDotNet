using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the MUSDB18 data loader.
/// </summary>
/// <remarks>
/// <para>
/// MUSDB18 is a music source separation benchmark with 150 full-length tracks (100 train / 50 test)
/// with isolated stems: vocals, drums, bass, and other. Audio is stereo 44.1kHz.
/// </para>
/// </remarks>
public sealed class Musdb18DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 44100.</summary>
    public int SampleRate { get; set; } = 44100;
    /// <summary>Segment duration in seconds (random segments are extracted from tracks). Default is 6.</summary>
    public double SegmentDurationSeconds { get; set; } = 6.0;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "Sample rate must be positive.");
        if (SegmentDurationSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(SegmentDurationSeconds), "Segment duration must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
