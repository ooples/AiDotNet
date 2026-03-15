using AiDotNet.Interfaces;
using AiDotNet.Classification.ImbalancedEnsemble;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class BalancedBaggingClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new BalancedBaggingClassifier<double>();
}
