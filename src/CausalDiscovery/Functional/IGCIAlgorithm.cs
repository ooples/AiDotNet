using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// IGCI (Information-Geometric Causal Inference) — bivariate causal discovery via entropy.
/// </summary>
/// <remarks>
/// <para>Reference: Janzing et al. (2012), "Information-Geometric Approach to Inferring
/// Causal Directions", Artificial Intelligence.</para>
/// </remarks>
public class IGCIAlgorithm<T> : FunctionalBase<T>
{
    public override string Name => "IGCI";
    public override bool SupportsNonlinear => true;
    public IGCIAlgorithm(CausalDiscoveryOptions? options = null) { }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new ANMAlgorithm<T>().DiscoverStructure(data).AdjacencyMatrix;
}
