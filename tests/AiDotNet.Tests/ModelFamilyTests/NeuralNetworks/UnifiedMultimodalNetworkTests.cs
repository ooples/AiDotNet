using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class UnifiedMultimodalNetworkTests : NeuralNetworkModelTestBase<float>
{
    // Default: inputSize=768 (embedding dim), outputSize=100
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [100];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new UnifiedMultimodalNetwork<float>();

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
