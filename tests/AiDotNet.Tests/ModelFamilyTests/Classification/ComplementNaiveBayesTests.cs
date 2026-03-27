using AiDotNet.Interfaces;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class ComplementNaiveBayesTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ComplementNaiveBayes<double>();

    // ComplementNB requires non-negative count features
    protected override (Matrix<double> X, Vector<double> Y) GenerateData(
        int samples, int features, int nClasses, Random rng)
        => ModelTestHelpers.GenerateCountClassificationData(samples, features, nClasses, rng);
}
