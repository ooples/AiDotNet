using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #2087: out of the HeavyTimeout lane. The model used to resolve its decoders as a chain (each fed
// the previous decoder's output), which inflated it past 500M parameters and broke serialization; it
// now resolves them from the shared embedding (~110M by default). Per the #1706 out-of-memory ladder
// the test also shrinks the model: a 256-wide embedding and two transformer layers. The four modality
// decoders and the classification head keep their default output widths.
public class UnifiedMultimodalNetworkTests : NeuralNetworkModelTestBase<float>
{
    private const int EmbeddingWidth = 256;

    protected override int[] InputShape => [EmbeddingWidth];
    protected override int[] OutputShape => [100];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new UnifiedMultimodalNetwork<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputSize: EmbeddingWidth,
                outputSize: 100),
            new UnifiedMultimodalNetworkOptions { EmbeddingDimension = EmbeddingWidth, NumTransformerLayers = 2 },
            seed: 42);
    // UnifiedMultimodalNetwork is a MultiClassClassification model trained with
    // CrossEntropyWithLogitsLoss, so its paper-faithful target is a one-hot class
    // label, not a continuous-uniform tensor. The test base documents this exact
    // override hook for "multi-class classification with cross-entropy" families
    // (see CreateRandomTargetTensor remarks). A continuous-uniform target is not a
    // valid class distribution: cross-entropy can never drive the prediction onto
    // it, so the MSE proxy never reaches zero and two independent random runs are
    // not comparable. With a one-hot target the softmax prediction converges to the
    // label (MSE -> 0), which is the correct objective for this model.
    protected override Tensor<float> CreateRandomTargetTensor(int[] shape, System.Random rng)
    {
        var target = new Tensor<float>(shape);
        // Pick a single active class along the last axis for each row.
        int numClasses = shape[^1];
        int rows = target.Length / numClasses;
        for (int r = 0; r < rows; r++)
            target[r * numClasses + rng.Next(numClasses)] = 1.0f;
        return target;
    }
}
