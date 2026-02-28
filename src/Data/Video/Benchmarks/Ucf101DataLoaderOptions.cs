using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Configuration options for the UCF101 data loader.
/// </summary>
/// <remarks>
/// <para>
/// UCF101 contains 13,320 video clips from 101 action categories. Videos are sourced from
/// YouTube with realistic camera motion and varying conditions.
/// </para>
/// </remarks>
public sealed class Ucf101DataLoaderOptions
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
    /// <summary>Split number (1, 2, or 3). Default is 1.</summary>
    public int SplitNumber { get; set; } = 1;
}
