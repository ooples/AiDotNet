using AiDotNet.Interfaces;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class ComplementNaiveBayesTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ComplementNaiveBayes<double>();
}
