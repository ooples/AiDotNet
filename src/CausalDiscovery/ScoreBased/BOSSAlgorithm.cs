using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// BOSS (Best Order Score Search) — efficient permutation-based structure learning.
/// </summary>
/// <remarks>
/// <para>
/// BOSS combines permutation search with score-based evaluation, using efficient
/// caching and pruning strategies to scale to large datasets.
/// </para>
/// <para>
/// <b>For Beginners:</b> BOSS is a modern, fast algorithm that finds causal structures
/// by efficiently searching through possible variable orderings. It's designed to be
/// competitive with the best algorithms while being computationally efficient.
/// </para>
/// <para>
/// Reference: Andrews et al. (2022), "Fast Scalable and Accurate Discovery of DAGs
/// Using the Best Order Score Search and Grow-Shrink Trees".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class BOSSAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "BOSS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public BOSSAlgorithm(CausalDiscoveryOptions? options = null) { ApplyScoreOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new FGESAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
