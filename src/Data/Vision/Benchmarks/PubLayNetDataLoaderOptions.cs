using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the PubLayNet document layout analysis data loader.
/// </summary>
/// <remarks>
/// <para>
/// PubLayNet contains ~360K document images with layout annotations (text, title, list, table, figure).
/// Standard benchmark for document layout analysis.
/// </para>
/// </remarks>
public sealed class PubLayNetDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Image width after resizing. Default is 224.</summary>
    public int ImageWidth { get; set; } = 224;
    /// <summary>Image height after resizing. Default is 224.</summary>
    public int ImageHeight { get; set; } = 224;
    /// <summary>Maximum number of layout regions per image. Default is 50.</summary>
    public int MaxRegions { get; set; } = 50;
    /// <summary>Number of layout classes (text, title, list, table, figure). Default is 5.</summary>
    public int NumClasses { get; set; } = 5;
}
