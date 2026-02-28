using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the AudioSet data loader.
/// </summary>
/// <remarks>
/// <para>
/// AudioSet is a large-scale audio event dataset with 2M+ 10-second YouTube clips
/// labeled with 527 audio event categories. Multi-label classification task.
/// Requires pre-downloaded audio (YouTube clips converted to WAV).
/// </para>
/// </remarks>
public sealed class AudioSetDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false (requires YouTube download).</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 16000.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Audio clip duration in seconds. Default is 10.</summary>
    public double ClipDurationSeconds { get; set; } = 10.0;
}
