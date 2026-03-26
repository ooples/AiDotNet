using AiDotNet.Interfaces;
using AiDotNet.Classification.Ordinal;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class OrdinalRidgeRegressionTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new OrdinalRidgeRegression<double>();
}
