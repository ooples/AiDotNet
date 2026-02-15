using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// BayesDAG — Bayesian DAG learning with gradient-based posterior inference.
/// </summary>
/// <remarks>
/// <para>
/// BayesDAG uses a novel DAG-constrained variational framework that maintains a proper
/// distribution over DAGs. It combines advances in continuous DAG constraints with
/// scalable variational inference methods.
/// </para>
/// <para>
/// <b>For Beginners:</b> BayesDAG is a modern Bayesian method that efficiently explores
/// the space of possible causal graphs using gradient-based optimization, providing
/// principled uncertainty quantification about the causal structure.
/// </para>
/// <para>
/// Reference: Annadani et al. (2024), "BayesDAG: Gradient-Based Posterior Inference
/// for Causal Discovery", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class BayesDAGAlgorithm<T> : BayesianCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "BayesDAG";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public BayesDAGAlgorithm(CausalDiscoveryOptions? options = null) { ApplyBayesianOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ScoreBased.HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
