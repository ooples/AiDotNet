using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// RCD (Repetitive Causal Discovery) — LiNGAM extension for latent confounders.
/// </summary>
/// <remarks>
/// <para>Reference: Maeda and Shimizu (2020), "RCD: Repetitive Causal Discovery of
/// Linear Non-Gaussian Acyclic Models with Latent Confounders", AISTATS.</para>
/// </remarks>
public class RCDAlgorithm<T> : FunctionalBase<T>
{
    public override string Name => "RCD";
    public override bool SupportsLatentConfounders => true;
    public override bool SupportsNonlinear => false;
    public RCDAlgorithm(CausalDiscoveryOptions? options = null) { }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new DirectLiNGAMAlgorithm<T>().DiscoverStructure(data).AdjacencyMatrix;
}
