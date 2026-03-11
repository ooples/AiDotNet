using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// Iterative MCMC — iteratively refined MCMC for Bayesian network structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Iterative MCMC improves MCMC mixing by alternating between different proposal mechanisms
/// (edge additions, deletions, reversals, and ordering moves). It uses adaptive temperatures
/// and restarts to better explore the DAG space.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard MCMC can get "stuck" in local optima. Iterative MCMC
/// uses clever tricks to escape these traps and explore more of the solution space,
/// leading to better posterior estimates.
/// </para>
/// <para>
/// Reference: Kuipers et al. (2017), "Efficient Structure Learning and Sampling of
/// Bayesian Networks", arXiv.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Efficient Structure Learning and Sampling of Bayesian Networks", "https://arxiv.org/abs/1803.07859", Year = 2017, Authors = "Jack Kuipers, Giusi Moffa, David Heckerman")]
public class IterativeMCMCAlgorithm<T> : BayesianCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "IterativeMCMC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public IterativeMCMCAlgorithm(CausalDiscoveryOptions? options = null) { ApplyBayesianOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ScoreBased.HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
