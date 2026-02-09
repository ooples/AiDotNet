using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the IMDB 50k sentiment analysis data loader.
/// </summary>
public sealed class Imdb50kDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum number of tokens (words) per review. Default is 256.</summary>
    public int MaxSequenceLength { get; set; } = 256;
    /// <summary>Maximum vocabulary size (most frequent words). Default is 10000.</summary>
    public int VocabularySize { get; set; } = 10000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
