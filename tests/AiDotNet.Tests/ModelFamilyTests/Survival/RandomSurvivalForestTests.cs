using AiDotNet.Interfaces;
using AiDotNet.SurvivalAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Survival;

/// <summary>
/// Manual factory for <see cref="RandomSurvivalForest{T}"/>. The auto-generator
/// routes survival models that also declare <c>[ModelTask(ModelTask.Regression)]</c>
/// to <see cref="RegressionModelTestBase"/> because Priority 15 (Regression task
/// + Matrix input) fires before Priority 17b (Survival). The Regression base's
/// invariants (R² ≥ 0 on linear data, coefficient signs, etc.) are not
/// meaningful for a forest of survival trees that predicts cumulative-hazard
/// estimates rather than linear point predictions, so we override here to use
/// the Survival-specific invariant suite.
/// </summary>
/// <remarks>
/// Per Ishwaran et al. 2008 ("Random Survival Forests"), the algorithm is an
/// ensemble of survival trees built on bootstrap samples that split using the
/// log-rank statistic. The paper's defaults are 100 trees, max depth ≈ 10,
/// min samples per leaf 6, and <c>maxFeatures = √p</c>. We use those defaults
/// (with <c>seed = 42</c> for reproducibility) — the SurvivalModelTestBase
/// invariants (deterministic predictions, finite output, parameters non-empty,
/// metadata after training, output dimension matches input rows) are cheap to
/// run even at the paper's full forest size on the 80-sample / 3-feature
/// smoke shape used by the base.
/// </remarks>
public class RandomSurvivalForestTests : SurvivalModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new RandomSurvivalForest<double>(seed: 42);
}
