using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class QuantileRegressionForestsTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new QuantileRegressionForests<double>();
}
