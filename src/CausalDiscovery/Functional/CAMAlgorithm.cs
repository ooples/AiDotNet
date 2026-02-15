using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CAM (Causal Additive Model) — order-based discovery with additive models.
/// </summary>
/// <remarks>
/// <para>Reference: Buhlmann et al. (2014), "CAM: Causal Additive Models, High-Dimensional
/// Order Search and Penalized Regression", Annals of Statistics.</para>
/// </remarks>
public class CAMAlgorithm<T> : FunctionalBase<T>
{
    public override string Name => "CAM";
    public override bool SupportsNonlinear => true;
    public CAMAlgorithm(CausalDiscoveryOptions? options = null) { }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new DirectLiNGAMAlgorithm<T>().DiscoverStructure(data).AdjacencyMatrix;
}
