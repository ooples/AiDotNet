using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GloVeTests : NeuralNetworkModelTestBase
{
    // GloVe.Predict / .Train output a rank-3 tensor [embeddingDim,
    // embeddingDim, 1] (vocabSize == embeddingDim² when both are 100 by
    // default), not the architecture.OutputSize=768 it advertises. The
    // shape used here lets MSE loss match the predicted-output rank in
    // Training_ShouldReduceLoss / GradientFlow_ShouldBeNonZeroAndFinite.
    protected override int[] InputShape => [1];
    protected override int[] OutputShape => [100, 100, 1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GloVe<double>();
}
