using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Information-Geometric Approach to Inferring Causal Directions", "https://doi.org/10.1016/j.artint.2012.01.002", Year = 2012, Authors = "Dominik Janzing, Joris Mooij, Kun Zhang, Jan Lemeire, Jakob Zscheischler, Povilas Daniusis, Bastian Steudel, Bernhard Scholkopf")]
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
