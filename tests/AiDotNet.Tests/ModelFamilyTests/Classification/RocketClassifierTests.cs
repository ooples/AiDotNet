using AiDotNet.Interfaces;
using AiDotNet.Classification.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class RocketClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new RocketClassifier<double>();

    // Rocket treats features as time steps — needs longer sequences for convolution kernels
    protected override int Features => 50;
    // Rocket uses random convolution kernels — more training data helps with feature extraction
    protected override int TrainSamples => 200;
    // Rocket uses random convolution kernels, no flat parameter vector
    protected override bool HasFlatParameters => false;
}
