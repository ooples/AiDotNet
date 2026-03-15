using AiDotNet.Interfaces;
using AiDotNet.Classification.Ordinal;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class OrdinalLogisticRegressionTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new OrdinalLogisticRegression<double>();
}
