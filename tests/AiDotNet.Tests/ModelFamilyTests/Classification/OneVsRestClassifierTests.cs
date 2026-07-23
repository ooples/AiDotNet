using AiDotNet.Interfaces;
using AiDotNet.Classification.Meta;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class OneVsRestClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new OneVsRestClassifier<double>();

    protected override bool HasFlatParameters => false;
}
