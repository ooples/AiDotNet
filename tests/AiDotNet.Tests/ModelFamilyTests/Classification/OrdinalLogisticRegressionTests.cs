using AiDotNet.Interfaces;
using AiDotNet.Classification.Ordinal;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

[Collection("NonParallelClassification")]
public class OrdinalLogisticRegressionTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new OrdinalLogisticRegression<double>();
}
