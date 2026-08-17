using AiDotNet.Interfaces;
using AiDotNet.Regression.MixedEffects;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class LinearMixedModelTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
    {
        var model = new LinearMixedModel<double>();
        // LMM requires at least one random effect. Use a random intercept
        // grouped by the first feature column.
        model.AddRandomIntercept("group", groupColumnIndex: 0);
        return model;
    }

    /// <summary>
    /// Column 0 is the grouping column, so it is a factor rather than a predictor and has no fixed
    /// effect of its own - the same convention lme4 uses for <c>y ~ x + (1|group)</c>.
    /// </summary>
    protected override ISet<int> StructuralFeatureIndices => new HashSet<int> { 0 };
}
