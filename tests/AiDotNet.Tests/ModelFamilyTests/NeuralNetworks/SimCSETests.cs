using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #2087 / #1706: out of the HeavyTimeout lane. At its default BERT-base scale (768 wide, 12 layers,
// 3072 FFN) the training invariants ran past the 120 s gate, so per the #1706 out-of-memory ladder
// the test shrinks the encoder to 128 wide, 2 layers, 2 heads and a 512 FFN. The vocabulary, the
// pooling and every head keep their defaults.
public class SimCSETests : NeuralNetworkModelTestBase<float>
{
    protected override bool RequiresHeavySerialization => true;

    // Per Gao et al. (2021): SimCSE outputs [CLS] embeddings of size embeddingDimension (768)
    protected override int[] InputShape => [1, Width];
    protected override int[] OutputShape => [1, Width];

    private const int Width = 128;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SimCSE<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: Width,
                outputSize: Width),
            new SimCSEOptions
            {
                EmbeddingDimension = Width,
                NumLayers = 2,
                NumHeads = 2,
                FeedForwardDim = 4 * Width,
            });
}
