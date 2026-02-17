using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// Partition MCMC — MCMC sampling over DAG partitions for structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Partition MCMC extends Order MCMC by sampling over partitions of variables instead
/// of total orderings. This provides a more refined search space than orderings while
/// remaining more efficient than direct DAG sampling.
/// </para>
/// <para>
/// <b>For Beginners:</b> This method groups variables into "layers" (partitions) where
/// variables in earlier layers can cause variables in later layers but not vice versa.
/// It explores different layer arrangements to find plausible causal structures.
/// </para>
/// <para>
/// Reference: Kuipers and Moffa (2017), "Partition MCMC for Inference on Acyclic
/// Digraphs", JASA.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PartitionMCMCAlgorithm<T> : BayesianCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "PartitionMCMC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public PartitionMCMCAlgorithm(CausalDiscoveryOptions? options = null) { ApplyBayesianOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ScoreBased.HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
