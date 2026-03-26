using AiDotNet.Interfaces;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class MultinomialNaiveBayesTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new MultinomialNaiveBayes<double>();

    // MultinomialNB requires non-negative count features (word frequencies, etc.)
    protected override (Matrix<double> X, Vector<double> Y) GenerateData(
        int samples, int features, int nClasses, Random rng)
        => ModelTestHelpers.GenerateCountClassificationData(samples, features, nClasses, rng);
}
