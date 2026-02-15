using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CCDr (Concave penalized Coordinate Descent with reparameterization).
/// </summary>
/// <remarks>
/// <para>
/// This is a baseline implementation that delegates to DirectLiNGAM.
/// A full CCDr implementation with concave penalty optimization is planned.
/// </para>
/// <para>Reference: Aragam and Zhou (2015), "Concave Penalized Estimation of Sparse
/// Gaussian Bayesian Networks", JMLR.</para>
/// </remarks>
internal class CCDrAlgorithm<T> : FunctionalBase<T>
{
    private readonly CausalDiscoveryOptions? _options;
    public override string Name => "CCDr";
    public override bool SupportsNonlinear => false;
    public CCDrAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _options = options;

    }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new DirectLiNGAMAlgorithm<T>(_options).DiscoverStructure(data).AdjacencyMatrix;
}
