using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// IGCI (Information-Geometric Causal Inference) — bivariate causal discovery via entropy.
/// </summary>
/// <remarks>
/// <para>
/// This is a baseline implementation that delegates to ANM.
/// A full IGCI implementation with entropy-based inference is planned.
/// </para>
/// <para>Reference: Janzing et al. (2012), "Information-Geometric Approach to Inferring
/// Causal Directions", Artificial Intelligence.</para>
/// </remarks>
public class IGCIAlgorithm<T> : FunctionalBase<T>
{
    private readonly CausalDiscoveryOptions? _options;
    public override string Name => "IGCI";
    public override bool SupportsNonlinear => false;
    public IGCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _options = options;

    }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new ANMAlgorithm<T>(_options).DiscoverStructure(data).AdjacencyMatrix;
}
