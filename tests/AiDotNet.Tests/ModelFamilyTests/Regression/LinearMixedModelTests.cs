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
}
