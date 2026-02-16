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
    /// Maximum number of iterations for the Leiden algorithm. Default: 10.
    /// </summary>
    public int? MaxIterations { get; set; }

    /// <summary>
    /// Resolution parameter controlling community granularity. Higher = smaller communities. Default: 1.0.
    /// </summary>
    public double? Resolution { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null (non-deterministic).
    /// </summary>
    public int? Seed { get; set; }

    internal int GetEffectiveMaxIterations() => MaxIterations ?? 10;
    internal double GetEffectiveResolution() => Resolution ?? 1.0;
}
