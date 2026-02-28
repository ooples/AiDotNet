using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Graph;

/// <summary>
/// Configuration options for the Wikidata5M knowledge graph data loader.
/// </summary>
/// <remarks>
/// <para>
/// Wikidata5M is a large-scale knowledge graph with ~5M entities and ~21M triplets.
/// Supports link prediction and entity classification tasks.
/// </para>
/// </remarks>
public sealed class Wikidata5mDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of triplets to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Entity embedding dimension. Default is 128.</summary>
    public int EmbeddingDimension { get; set; } = 128;
}
