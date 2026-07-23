using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Graph;

/// <summary>
/// Configuration options for the temporal graph data loader.
/// </summary>
/// <remarks>
/// <para>
/// Temporal graphs model evolving networks with timestamped edges.
/// Supports datasets like Wikipedia edits and Reddit posts for dynamic link prediction.
/// </para>
/// </remarks>
public sealed class TemporalGraphDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of interactions to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Number of node feature dimensions. Default is 172.</summary>
    public int NodeFeatureDimension { get; set; } = 172;
    /// <summary>Number of edge feature dimensions. Default is 172.</summary>
    public int EdgeFeatureDimension { get; set; } = 172;
}
