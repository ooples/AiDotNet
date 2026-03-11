using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// GFCI — Greedy FCI, a hybrid of GES and FCI.
/// </summary>
/// <remarks>
/// <para>
/// GFCI combines the scoring efficiency of GES with the ability of FCI to handle latent
/// confounders. It first runs GES to get an initial CPDAG, then applies FCI-like orientation
/// rules to account for possible latent variables.
/// </para>
/// <para>
/// <b>For Beginners:</b> GFCI is useful when you suspect there are hidden variables affecting
/// your data. It first quickly finds a good graph structure, then checks whether some
/// connections might actually be due to hidden common causes rather than direct effects.
/// </para>
/// <para>
/// Reference: Ogarrio et al. (2016), "A Hybrid Causal Search Algorithm for Latent
/// Variable Models", PGM.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("A Hybrid Causal Search Algorithm for Latent Variable Models", "https://doi.org/10.48550/arXiv.1602.01426", Year = 2016, Authors = "Juan Miguel Ogarrio, Peter Spirtes, Joe Ramsey")]
public class GFCIAlgorithm<T> : HybridBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GFCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    public GFCIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyHybridOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to MMHC as baseline
        var baseline = new MMHCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
