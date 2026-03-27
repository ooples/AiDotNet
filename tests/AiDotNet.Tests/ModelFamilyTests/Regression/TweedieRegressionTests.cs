using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class TweedieRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new TweedieRegression<double>();
}
