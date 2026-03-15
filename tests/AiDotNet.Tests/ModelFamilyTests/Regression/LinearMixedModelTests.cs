using AiDotNet.Interfaces;
using AiDotNet.Regression.MixedEffects;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class LinearMixedModelTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new LinearMixedModel<double>();
}
