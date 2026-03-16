using AiDotNet.Interfaces;
using AiDotNet.Classification.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class MiniRocketClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new MiniRocketClassifier<double>();

    // MiniRocket treats features as time steps — needs longer sequences
    protected override int Features => 50;
    protected override int TrainSamples => 200;
    // MiniRocket uses random convolution kernels, no flat parameter vector
    protected override bool HasFlatParameters => false;
}
