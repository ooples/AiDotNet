using AiDotNet.Interfaces;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class ExtraTreesClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ExtraTreesClassifier<double>();
}
