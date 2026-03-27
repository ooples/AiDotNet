using AiDotNet.Interfaces;
using AiDotNet.Classification.DiscriminantAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class LinearDiscriminantAnalysisTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new LinearDiscriminantAnalysis<double>();
}
