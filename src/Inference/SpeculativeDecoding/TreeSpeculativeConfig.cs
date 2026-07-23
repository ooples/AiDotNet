namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Configuration for tree-based speculative decoding.
/// </summary>
internal class TreeSpeculativeConfig
{
    /// <summary>Number of branches per node.</summary>
    public int BranchFactor { get; set; } = 2;

    /// <summary>Maximum tree depth.</summary>
    public int MaxDepth { get; set; } = 4;

    /// <summary>Maximum total nodes in tree.</summary>
    public int MaxNodes { get; set; } = 16;

    /// <summary>Random seed.</summary>
    public int? Seed { get; set; }
}
