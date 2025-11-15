namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Defines the level of optimization to apply to the computation graph.
/// Higher levels apply more aggressive optimizations but may take longer to compile.
/// </summary>
public enum OptimizationLevel
{
    /// <summary>
    /// No optimization - use the graph as-is.
    /// </summary>
    None = 0,

    /// <summary>
    /// Basic optimizations - dead code elimination, constant folding.
    /// Fast to compile, minimal speedup.
    /// </summary>
    Basic = 1,

    /// <summary>
    /// Standard optimizations - includes basic + operator fusion + algebraic simplification.
    /// Balanced compile time and performance. Recommended for most use cases.
    /// </summary>
    Standard = 2,

    /// <summary>
    /// Aggressive optimizations - includes standard + memory optimizations + CSE.
    /// Longer compile time, significant speedup. Good for production deployments.
    /// </summary>
    Aggressive = 3,

    /// <summary>
    /// Maximum optimizations - all available optimizations.
    /// Longest compile time, maximum speedup. Use for critical inference paths.
    /// </summary>
    Maximum = 4
}
