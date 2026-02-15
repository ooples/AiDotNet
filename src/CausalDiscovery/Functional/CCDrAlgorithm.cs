using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CCDr (Concave penalized Coordinate Descent with reparameterization).
/// </summary>
/// <remarks>
/// <para>Reference: Aragam and Zhou (2015), "Concave Penalized Estimation of Sparse
/// Gaussian Bayesian Networks", JMLR.</para>
/// </remarks>
public class CCDrAlgorithm<T> : FunctionalBase<T>
{
    public override string Name => "CCDr";
    public override bool SupportsNonlinear => false;
    public CCDrAlgorithm(CausalDiscoveryOptions? options = null) { }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new DirectLiNGAMAlgorithm<T>().DiscoverStructure(data).AdjacencyMatrix;
}
