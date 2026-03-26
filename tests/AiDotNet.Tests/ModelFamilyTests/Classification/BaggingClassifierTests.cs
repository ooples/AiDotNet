using AiDotNet.Interfaces;
using AiDotNet.Classification.Meta;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class BaggingClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new BaggingClassifier<double>();

    protected override bool HasFlatParameters => false;
}
