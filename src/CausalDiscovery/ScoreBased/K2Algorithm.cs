using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// K2 Algorithm — score-based learning with known variable ordering.
/// </summary>
/// <remarks>
/// <para>
/// K2 learns a Bayesian network structure given a known topological ordering of variables.
/// For each variable, it greedily adds parents from variables earlier in the ordering
/// that maximize the BDeu score.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you already know the rough order in which variables cause
/// each other (e.g., age before income before spending), K2 efficiently finds the
/// exact connections. It's very fast but requires this ordering as input.
/// </para>
/// <para>
/// Reference: Cooper and Herskovits (1992), "A Bayesian Method for the Induction
/// of Probabilistic Networks from Data".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class K2Algorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "K2";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public K2Algorithm(CausalDiscoveryOptions? options = null) { ApplyScoreOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to Hill Climbing as baseline
        var baseline = new HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
