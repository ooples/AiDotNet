using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Configuration options for the Kinetics-400 data loader.
/// </summary>
/// <remarks>
/// <para>
/// Kinetics-400 contains ~300K 10-second video clips covering 400 human action classes.
/// Videos are sourced from YouTube. Requires pre-extracted frames or video files.
/// </para>
/// </remarks>
public sealed class Kinetics400DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Number of frames to sample per video. Default is 16.</summary>
    public int FramesPerVideo { get; set; } = 16;
    /// <summary>Frame width. Default is 224.</summary>
    public int FrameWidth { get; set; } = 224;
    /// <summary>Frame height. Default is 224.</summary>
    public int FrameHeight { get; set; } = 224;
    /// <summary>Whether to normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;
}
