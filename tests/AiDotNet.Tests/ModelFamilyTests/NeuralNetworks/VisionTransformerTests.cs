using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VisionTransformerTests : NeuralNetworkModelTestBase<float>
{
    // ViT default architecture is [batch, 3, 224, 224] (paper-faithful
    // ImageNet-1k geometry). The parameterless ctor hard-codes this and
    // rejects any other resolution at Predict, so the test base must
    // submit a 224x224 input to match.
    protected override int[] InputShape => [1, 3, 224, 224];
    protected override int[] OutputShape => [1000];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new VisionTransformer<float>();
}
