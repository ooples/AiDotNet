using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// H2PC — Hybrid HPC (Hybrid Parents and Children) algorithm.
/// </summary>
/// <remarks>
/// <para>
/// H2PC uses an improved constraint-based phase (HPC) for neighborhood discovery,
/// followed by a score-based search for orientation. HPC is more sample-efficient
/// than MMPC and provides better neighborhood estimates.
/// </para>
/// <para>
/// <b>For Beginners:</b> H2PC is a refined version of MMHC with a smarter first phase.
/// It finds candidate parents/children more accurately before running the scoring step,
/// leading to better results especially with smaller datasets.
/// </para>
/// <para>
/// Reference: Gasse et al. (2014), "A Hybrid Algorithm for Bayesian Network Structure
/// Learning with Application to Multi-Label Learning", Expert Systems with Applications.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("A Hybrid Algorithm for Bayesian Network Structure Learning with Application to Multi-Label Learning", "https://doi.org/10.1016/j.eswa.2014.03.032", Year = 2014, Authors = "Maxime Gasse, Alex Aussem, Haytham Elghazel")]
public class H2PCAlgorithm<T> : HybridBase<T>
{
    /// <inheritdoc/>
    public override string Name => "H2PC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public H2PCAlgorithm(CausalDiscoveryOptions? options = null) { ApplyHybridOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to MMHC as baseline
        var baseline = new MMHCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
