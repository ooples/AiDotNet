using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// RCD (Repetitive Causal Discovery) — LiNGAM extension for latent confounders.
/// </summary>
/// <remarks>
/// <para>
/// This is a baseline implementation that delegates to DirectLiNGAM.
/// A full RCD implementation with repetitive latent confounder detection is planned.
/// </para>
/// <para>Reference: Maeda and Shimizu (2020), "RCD: Repetitive Causal Discovery of
/// Linear Non-Gaussian Acyclic Models with Latent Confounders", AISTATS.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("RCD: Repetitive Causal Discovery of Linear Non-Gaussian Acyclic Models with Latent Confounders", "https://proceedings.mlr.press/v108/maeda20a.html", Year = 2020, Authors = "Takashi Nicholas Maeda, Shohei Shimizu")]
public class RCDAlgorithm<T> : FunctionalBase<T>
{
    private readonly CausalDiscoveryOptions? _options;
    public override string Name => "RCD";
    public override bool SupportsLatentConfounders => true;
    public override bool SupportsNonlinear => false;
    public RCDAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _options = options;

    }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new DirectLiNGAMAlgorithm<T>(_options).DiscoverStructure(data).AdjacencyMatrix;
}
