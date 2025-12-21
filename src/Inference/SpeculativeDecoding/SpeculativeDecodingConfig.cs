using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Configuration for speculative decoding.
/// </summary>
/// <typeparam name="T">The numeric type for threshold values.</typeparam>
internal class SpeculativeDecodingConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Number of draft tokens to generate per verification.
    /// </summary>
    public int NumDraftTokens { get; set; } = 5;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Whether to use tree-based speculation (multiple draft continuations).
    /// </summary>
    public bool UseTreeSpeculation { get; set; } = false;

    /// <summary>
    /// Branching factor for tree speculation.
    /// </summary>
    public int TreeBranchFactor { get; set; } = 2;

    /// <summary>
    /// Maximum tree depth for tree speculation.
    /// </summary>
    public int MaxTreeDepth { get; set; } = 4;

    /// <summary>
    /// Minimum acceptance rate before reducing draft length.
    /// </summary>
    public T MinAcceptanceRate { get; set; } = NumOps.FromDouble(0.5);

    /// <summary>
    /// Whether to dynamically adjust draft length based on acceptance rate.
    /// </summary>
    public bool AdaptiveDraftLength { get; set; } = false;
}
