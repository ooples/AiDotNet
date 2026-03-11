using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Concave Penalized Estimation of Sparse Gaussian Bayesian Networks", "https://jmlr.org/papers/v16/aragam15a.html", Year = 2015, Authors = "Bryon Aragam, Qing Zhou")]
public class CCDrAlgorithm<T> : FunctionalBase<T>
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
