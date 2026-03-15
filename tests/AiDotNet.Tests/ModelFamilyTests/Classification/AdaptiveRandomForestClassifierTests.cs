using AiDotNet.Interfaces;
using AiDotNet.Classification.Online;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class AdaptiveRandomForestClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new AdaptiveRandomForestClassifier<double>();
}
