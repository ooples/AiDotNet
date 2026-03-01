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

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (BufferSize <= 0) throw new ArgumentOutOfRangeException(nameof(BufferSize), "BufferSize must be positive.");
        if (Weights != null)
        {
            if (Weights.Length == 0) throw new ArgumentOutOfRangeException(nameof(Weights), "Weights array must not be empty when specified.");
            for (int i = 0; i < Weights.Length; i++)
                if (Weights[i] < 0) throw new ArgumentOutOfRangeException(nameof(Weights), $"Weight at index {i} must be non-negative.");
        }
    }
}
