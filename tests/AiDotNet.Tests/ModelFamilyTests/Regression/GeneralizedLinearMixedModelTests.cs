using AiDotNet.Interfaces;
using AiDotNet.Regression.MixedEffects;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class GeneralizedLinearMixedModelTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
    {
        var model = new GeneralizedLinearMixedModel<double>();
        // GLMM requires at least one random effect. Use a random intercept
        // grouped by the first feature column (column 0 treated as group indicator).
        model.AddRandomIntercept("group", groupColumnIndex: 0);
        return model;
    }
}
