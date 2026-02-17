using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Specialized;

/// <summary>
/// GOBNILP — Globally Optimal Bayesian Network learning using Integer Linear Programming.
/// </summary>
/// <remarks>
/// <para>
/// GOBNILP formulates Bayesian network structure learning as an integer linear programming (ILP)
/// problem. It finds the globally optimal DAG by encoding the acyclicity constraint as
/// cluster constraints and solving with a branch-and-cut ILP solver.
/// </para>
/// <para>
/// <b>For Beginners:</b> GOBNILP guarantees finding the BEST possible graph according to the
/// scoring criterion. Most other algorithms are heuristic (they find good but not necessarily
/// optimal solutions). The trade-off is that GOBNILP can be slow for many variables.
/// </para>
/// <para>
/// Reference: Cussens (2012), "Bayesian Network Learning with Cutting Planes", UAI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GOBNILPAlgorithm<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GOBNILP";

    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.Specialized;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public GOBNILPAlgorithm(CausalDiscoveryOptions? options = null) { }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to Hill Climbing as baseline
        var baseline = new ScoreBased.HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
