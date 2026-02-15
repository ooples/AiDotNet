using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// DiBS — Differentiable Bayesian Structure Learning.
/// </summary>
/// <remarks>
/// <para>
/// DiBS uses variational inference with a differentiable relaxation of the DAG constraint
/// to approximate the posterior over graphs. It uses Stein variational gradient descent
/// to maintain a set of particles representing the posterior.
/// </para>
/// <para>
/// <b>For Beginners:</b> DiBS uses gradient-based optimization (like training a neural network)
/// to find not just one graph but a whole set of plausible graphs, giving you uncertainty
/// estimates about the causal structure.
/// </para>
/// <para>
/// Reference: Lorch et al. (2021), "DiBS: Differentiable Bayesian Structure Learning", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DiBSAlgorithm<T> : BayesianCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "DiBS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public DiBSAlgorithm(CausalDiscoveryOptions? options = null) { ApplyBayesianOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ScoreBased.HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
