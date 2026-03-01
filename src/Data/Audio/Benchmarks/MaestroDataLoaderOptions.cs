using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the MAESTRO data loader.
/// </summary>
/// <remarks>
/// <para>
/// MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) contains ~200 hours
/// of piano performances with aligned MIDI and audio (WAV, 44.1kHz stereo).
/// Used for piano transcription and music generation tasks.
/// </para>
/// </remarks>
public sealed class MaestroDataLoaderOptions
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
    /// <summary>Maximum audio duration in seconds. Default is 10.</summary>
    public double MaxDurationSeconds { get; set; } = 10.0;
    /// <summary>MAESTRO version. Default is "v3.0.0".</summary>
    public string Version { get; set; } = "v3.0.0";

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "SampleRate must be positive.");
        if (MaxDurationSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(MaxDurationSeconds), "MaxDurationSeconds must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
