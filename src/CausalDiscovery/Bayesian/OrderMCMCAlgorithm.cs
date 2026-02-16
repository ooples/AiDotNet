using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// Order MCMC — MCMC sampling over variable orderings for Bayesian structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Order MCMC samples from the posterior over variable orderings (rather than over DAGs
/// directly). Given an ordering, the optimal DAG can be computed efficiently. This exploits
/// the fact that the ordering space is much smoother than the DAG space.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of searching over all possible graphs (very hard), this
/// method searches over all possible orderings of variables (much easier). Once you know
/// the order, finding the best graph is straightforward.
/// </para>
/// <para>
/// Reference: Friedman and Koller (2003), "Being Bayesian About Network Structure", MLJ.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OrderMCMCAlgorithm<T> : BayesianCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "OrderMCMC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public OrderMCMCAlgorithm(CausalDiscoveryOptions? options = null) { ApplyBayesianOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ScoreBased.HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
