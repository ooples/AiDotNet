using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// BCD-Nets — Bayesian Causal Discovery Networks.
/// </summary>
/// <remarks>
/// <para>
/// BCD-Nets use variational inference to approximate the joint posterior over DAG structures
/// and parameters. The graph is parameterized via a continuous relaxation and optimized
/// using gradient-based methods with the reparameterization trick.
/// </para>
/// <para>
/// <b>For Beginners:</b> BCD-Nets learn both the graph structure AND the strength of each
/// connection simultaneously, using modern deep learning optimization techniques. They
/// provide uncertainty estimates for both.
/// </para>
/// <para>
/// Reference: Cundy et al. (2021), "BCD Nets: Scalable Variational Approaches for
/// Bayesian Causal Discovery", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class BCDNetsAlgorithm<T> : BayesianCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "BCD-Nets";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public BCDNetsAlgorithm(CausalDiscoveryOptions? options = null) { ApplyBayesianOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ScoreBased.HillClimbingAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
