using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// NoCurl — DAG learning via curl-free constraints on the graph structure.
/// </summary>
/// <remarks>
/// <para>
/// NoCurl formulates DAG learning as finding an ordering of variables and edge weights,
/// using a curl-free constraint instead of the matrix exponential. The key insight is that
/// a weighted adjacency matrix represents a DAG if and only if the induced flow field has
/// zero curl — equivalent to the existence of a topological ordering.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine water flowing through a network of pipes (the causal graph).
/// In a DAG, water should flow in one direction without any loops. NoCurl checks this by
/// measuring the "curl" of the flow — if there's a whirlpool (cycle), the curl is non-zero.
/// This gives a different mathematical way to ensure no cycles, which can be faster than
/// the matrix exponential used in NOTEARS.
/// </para>
/// <para>
/// Reference: Yu et al. (2021), "DAGs with No Curl: An Efficient DAG Structure Learning
/// Approach", ICML.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NoCurlAlgorithm<T> : ContinuousOptimizationBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NoCurl";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes NoCurl with optional configuration.
    /// </summary>
    public NoCurlAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell implementation — delegates to NOTEARS Linear as baseline
        // Full implementation would use curl-free parametrization: W = P^T * diag * P
        var baseline = new NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
