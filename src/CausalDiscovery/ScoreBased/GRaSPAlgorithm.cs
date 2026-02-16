using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// GRaSP (Greedy Relaxation of Sparsest Permutation) — permutation-based causal discovery.
/// </summary>
/// <remarks>
/// <para>
/// GRaSP searches over permutations (orderings) of variables and selects the sparsest
/// DAG consistent with each ordering. It uses a greedy relaxation to efficiently explore
/// the permutation space.
/// </para>
/// <para>
/// <b>For Beginners:</b> GRaSP tries different orderings of variables and for each ordering,
/// finds the simplest (sparsest) causal graph. It's designed to find graphs with fewer
/// edges, which often corresponds to the true causal structure.
/// </para>
/// <para>
/// Reference: Lam et al. (2022), "Greedy Relaxations of the Sparsest Permutation Algorithm".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GRaSPAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GRaSP";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public GRaSPAlgorithm(CausalDiscoveryOptions? options = null) { ApplyScoreOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new GESAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
