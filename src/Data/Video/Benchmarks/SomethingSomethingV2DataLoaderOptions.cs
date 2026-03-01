using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Configuration options for the Something-Something V2 data loader.
/// </summary>
/// <remarks>
/// <para>
/// Something-Something V2 contains 220K video clips of humans performing 174 pre-defined
/// actions with everyday objects. Temporal reasoning is required to distinguish actions.
/// </para>
/// </remarks>
public sealed class SomethingSomethingV2DataLoaderOptions
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
