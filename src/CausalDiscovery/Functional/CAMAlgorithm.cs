using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CAM (Causal Additive Model) — order-based discovery with additive models.
/// </summary>
/// <remarks>
/// <para>
/// This is a baseline implementation that delegates to DirectLiNGAM.
/// A full CAM implementation with nonparametric regression scoring is planned.
/// </para>
/// <para>Reference: Buhlmann et al. (2014), "CAM: Causal Additive Models, High-Dimensional
/// Order Search and Penalized Regression", Annals of Statistics.</para>
/// </remarks>
internal class CAMAlgorithm<T> : FunctionalBase<T>
{
    private readonly CausalDiscoveryOptions? _options;
    public override string Name => "CAM";
    public override bool SupportsNonlinear => false;
    public CAMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _options = options;

    }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new DirectLiNGAMAlgorithm<T>(_options).DiscoverStructure(data).AdjacencyMatrix;
}
