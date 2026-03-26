using AiDotNet.Interfaces;
using AiDotNet.Classification.Boosting;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class HistGradientBoostingClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new HistGradientBoostingClassifier<double>();
}
