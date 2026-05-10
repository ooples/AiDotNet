using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Describable Textures Dataset (DTD) loader (Cimpoi et al. 2014).
/// </summary>
/// <remarks>
/// <para>
/// DTD — 47 texture classes × 120 images each (5,640 total). The canonical
/// texture-classification benchmark. Ten predefined train/val/test splits;
/// this loader uses split #1 by default (matches the standard reporting
/// convention). Image-list-based labels via <c>labels/{train,val,test}1.txt</c>.
/// </para>
/// </remarks>
public sealed class DtdDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    public int ImageSize { get; set; } = 224;
    public bool Normalize { get; set; } = true;
    /// <summary>Predefined split index (1..10). Default 1.</summary>
    public int SplitIndex { get; set; } = 1;
    public int? MaxSamples { get; set; }
}
