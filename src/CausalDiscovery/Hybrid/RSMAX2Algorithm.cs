using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// RSMAX2 — Restricted Maximization, a hybrid constraint-based + score-based algorithm.
/// </summary>
/// <remarks>
/// <para>
/// RSMAX2 uses any constraint-based algorithm to learn a skeleton (restrict the search space),
/// then maximizes a network score (BIC/BDeu) within the restricted space. It generalizes
/// MMHC by allowing any constraint algorithm in the first phase.
/// </para>
/// <para>
/// <b>For Beginners:</b> RSMAX2 is a general framework for combining any "candidate finder"
/// with any "best structure finder." MMHC is a specific instance of RSMAX2 that uses MMPC
/// for candidates and hill climbing for scoring.
/// </para>
/// <para>
/// Reference: Scutari (2010), "Learning Bayesian Networks with the bnlearn R Package", JSS.
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
[ModelPaper("Learning Bayesian Networks with the bnlearn R Package", "https://doi.org/10.18637/jss.v035.i03", Year = 2010, Authors = "Marco Scutari")]
public class RSMAX2Algorithm<T> : HybridBase<T>
{
    /// <inheritdoc/>
    public override string Name => "RSMAX2";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public RSMAX2Algorithm(CausalDiscoveryOptions? options = null) { ApplyHybridOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to MMHC as baseline
        var baseline = new MMHCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
