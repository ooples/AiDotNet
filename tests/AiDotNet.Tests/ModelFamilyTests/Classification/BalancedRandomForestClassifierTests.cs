using AiDotNet.Interfaces;
using AiDotNet.Classification.ImbalancedEnsemble;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class BalancedRandomForestClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new BalancedRandomForestClassifier<double>();
}
