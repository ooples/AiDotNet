using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CAM-UV — Causal Additive Model with Unobserved Variables.
/// </summary>
/// <remarks>
/// <para>
/// This is a baseline implementation that delegates to ANM.
/// A full CAM-UV implementation with latent variable handling is planned.
/// </para>
/// <para>Reference: Maeda and Shimizu (2021), "Causal Additive Models with Unobserved Variables".</para>
/// </remarks>
internal class CAMUVAlgorithm<T> : FunctionalBase<T>
{
    private readonly CausalDiscoveryOptions? _options;
    public override string Name => "CAM-UV";
    public override bool SupportsLatentConfounders => false;
    public override bool SupportsNonlinear => false;
    public CAMUVAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _options = options;

    }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new ANMAlgorithm<T>(_options).DiscoverStructure(data).AdjacencyMatrix;
}
