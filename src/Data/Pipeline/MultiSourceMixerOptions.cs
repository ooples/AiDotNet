namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Configuration options for multi-source data mixing.
/// </summary>
public sealed class MultiSourceMixerOptions
{
    /// <summary>Mixing weights for each data source (normalized internally). Default is equal weighting.</summary>
    public double[]? Weights { get; set; }
    /// <summary>Random seed for reproducibility. Default is null (random).</summary>
    public int? Seed { get; set; }
    /// <summary>Whether to stop when the smallest source is exhausted. Default is false (cycle smaller sources).</summary>
    public bool StopOnShortestSource { get; set; } = false;
    /// <summary>Buffer size for interleaving. Default is 1000.</summary>
    public int BufferSize { get; set; } = 1000;
}
