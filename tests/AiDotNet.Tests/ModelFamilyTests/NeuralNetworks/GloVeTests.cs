using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GloVeTests : NeuralNetworkModelTestBase
{
    // GloVe.Predict is just Forward(input) — it iterates Layers[0..3]
    // sequentially (W → W_tilde → b → b_tilde) and returns the last layer's
    // output. With InputShape=[1] (seqLen=1), the chained Dense layers produce
    // a rank-2 [seqLen=1, 1] tensor (b_tilde projects to a single bias
    // dimension). This shape lets MSE loss match the predicted-output rank in
    // Training_ShouldReduceLoss / GradientFlow_ShouldBeNonZeroAndFinite.
    protected override int[] InputShape => [1];
    protected override int[] OutputShape => [1, 1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GloVe<double>();
}
