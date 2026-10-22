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
[Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class InstructorEmbeddingTests : EmbeddingModelTestBase<float>
{
    // InstructorEmbedding's default ctor wires a 768-dim transformer
    // (inputSize=768, outputSize=768). The test base defaults to [1, 4]
    // input and [1, 1] target, which caused the loss computation to try
    // subtracting a [1, 768] prediction from a [1, 1] target and throw
    // "Tensor shapes must match. Got [1, 768] and [1, 1]." in
    // MeanSquaredErrorLoss.ComputeTapeLoss. Align the test shapes with
    // the model's actual input/output dimensions.
    protected override int[] InputShape => [1, Width];
    protected override int[] OutputShape => [1, Width];

    private const int Width = 128;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new InstructorEmbedding<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: Width,
                outputSize: Width),
            new InstructorEmbeddingOptions
            {
                EmbeddingDimension = Width,
                NumLayers = 2,
                NumHeads = 2,
                FeedForwardDim = 4 * Width,
            });
}
