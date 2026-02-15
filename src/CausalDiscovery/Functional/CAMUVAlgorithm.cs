using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CAM-UV — Causal Additive Model with Unobserved Variables.
/// </summary>
/// <remarks>
/// <para>Reference: Maeda and Shimizu (2021), "Causal Additive Models with Unobserved Variables".</para>
/// </remarks>
public class CAMUVAlgorithm<T> : FunctionalBase<T>
{
    public override string Name => "CAM-UV";
    public override bool SupportsLatentConfounders => true;
    public override bool SupportsNonlinear => true;
    public CAMUVAlgorithm(CausalDiscoveryOptions? options = null) { }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new ANMAlgorithm<T>().DiscoverStructure(data).AdjacencyMatrix;
}
