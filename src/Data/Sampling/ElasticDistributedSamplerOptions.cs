namespace AiDotNet.Data.Sampling;

/// <summary>
/// Configuration options for elastic distributed sampling.
/// </summary>
public sealed class ElasticDistributedSamplerOptions
{
    /// <summary>Total number of samples in the dataset. Required.</summary>
    public int DatasetSize { get; set; }
    /// <summary>Number of distributed workers (replicas). Default is 1.</summary>
    public int NumReplicas { get; set; } = 1;
    /// <summary>Rank of the current worker (0-based). Default is 0.</summary>
    public int Rank { get; set; } = 0;
    /// <summary>Whether to shuffle indices each epoch. Default is true.</summary>
    public bool Shuffle { get; set; } = true;
    /// <summary>Whether to pad the last batch so all replicas get the same number of samples. Default is true.</summary>
    public bool DropLast { get; set; } = true;
    /// <summary>Random seed for reproducibility. Default is null (random).</summary>
    public int? Seed { get; set; }
}
