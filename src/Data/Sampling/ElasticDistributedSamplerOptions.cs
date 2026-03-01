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
    /// <summary>Whether to drop the remainder so all replicas get exactly DatasetSize/NumReplicas samples. Default is true.</summary>
    public bool DropLast { get; set; } = true;
    /// <summary>Random seed for reproducibility. Default is null (random).</summary>
    public int? Seed { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (DatasetSize <= 0) throw new ArgumentOutOfRangeException(nameof(DatasetSize), "DatasetSize must be positive.");
        if (NumReplicas <= 0) throw new ArgumentOutOfRangeException(nameof(NumReplicas), "NumReplicas must be positive.");
        if (Rank < 0 || Rank >= NumReplicas) throw new ArgumentOutOfRangeException(nameof(Rank), "Rank must be in [0, NumReplicas).");
    }
}
