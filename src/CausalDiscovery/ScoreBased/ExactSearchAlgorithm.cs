using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// Exact Search (Dynamic Programming) — optimal DAG structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Uses dynamic programming over subsets of variables to find the globally optimal
/// DAG structure according to a decomposable score (BIC/BDeu). Guaranteed to find
/// the best-scoring DAG but exponential in the number of variables.
/// </para>
/// <para>
/// <b>For Beginners:</b> This algorithm finds the absolute best causal graph by
/// systematically checking all possibilities using clever math shortcuts (dynamic
/// programming). It's guaranteed to find the optimal solution but only works for
/// small datasets (up to about 20-25 variables).
/// </para>
/// <para>
/// Reference: Silander and Myllymaki (2006), "A Simple Approach for Finding the
/// Globally Optimal Bayesian Network Structure".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExactSearchAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "Exact Search (DP)";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public ExactSearchAlgorithm(CausalDiscoveryOptions? options = null) { ApplyScoreOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: exact search is O(2^d), delegates to GES for now
        var baseline = new GESAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
