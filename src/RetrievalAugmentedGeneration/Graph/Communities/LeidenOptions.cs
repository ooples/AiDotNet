using System;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

/// <summary>
/// Configuration options for the Leiden community detection algorithm.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Leiden algorithm finds groups (communities) of tightly connected nodes.
/// - Resolution: Higher values find smaller, more fine-grained communities. Lower values merge into larger groups.
/// - MaxIterations: How many times the algorithm refines the communities. More iterations = better quality but slower.
/// </para>
/// </remarks>
public class LeidenOptions
{
    /// <summary>
    /// Maximum number of iterations for the Leiden algorithm. Default: 10. Must be positive.
    /// </summary>
    public int? MaxIterations { get; set; }

    /// <summary>
    /// Resolution parameter controlling community granularity. Higher = smaller communities.
    /// Default: 1.0. Must be a finite positive number.
    /// </summary>
    public double? Resolution { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null (non-deterministic).
    /// </summary>
    public int? Seed { get; set; }

    internal int GetEffectiveMaxIterations()
    {
        int value = MaxIterations ?? 10;
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MaxIterations),
                value,
                "MaxIterations must be a positive integer.");
        }

        return value;
    }

    internal double GetEffectiveResolution()
    {
        double value = Resolution ?? 1.0;
        if (value <= 0.0 || double.IsNaN(value) || double.IsInfinity(value))
        {
            throw new ArgumentOutOfRangeException(
                nameof(Resolution),
                value,
                "Resolution must be a finite positive number.");
        }

        return value;
    }
}
