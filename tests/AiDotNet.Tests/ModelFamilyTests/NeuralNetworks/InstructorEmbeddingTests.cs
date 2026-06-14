using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class InstructorEmbeddingTests : EmbeddingModelTestBase<float>
{
    // InstructorEmbedding's default ctor wires a 768-dim transformer
    // (inputSize=768, outputSize=768). The test base defaults to [1, 4]
    // input and [1, 1] target, which caused the loss computation to try
    // subtracting a [1, 768] prediction from a [1, 1] target and throw
    // "Tensor shapes must match. Got [1, 768] and [1, 1]." in
    // MeanSquaredErrorLoss.ComputeTapeLoss. Align the test shapes with
    // the model's actual input/output dimensions.
    protected override int[] InputShape => [1, 768];
    protected override int[] OutputShape => [1, 768];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new InstructorEmbedding<float>();
}
