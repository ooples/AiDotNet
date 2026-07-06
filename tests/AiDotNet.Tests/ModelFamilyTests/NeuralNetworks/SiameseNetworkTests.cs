using System;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SiameseNetworkTests : NeuralNetworkModelTestBase<float>
{
    // SiameseNetwork needs [batch, 2, features] input for pair comparison
    // Default: inputSize=128, outputSize=1 (similarity score)
    protected override int[] InputShape => [1, 2, 128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SiameseNetwork<float>();

    // Builds a pair tensor [1, 2, features] whose two halves are filled with
    // DISTINCT values, i.e. a genuine pair of two different items. The base
    // helpers fill every element with one constant, which for a pair-comparison
    // network produces a pair of two IDENTICAL items.
    private AiDotNet.Tensors.LinearAlgebra.Tensor<float> MakePair(float a, float b)
    {
        var t = CreateConstantTensor(InputShape, a);
        int half = t.Length / 2;
        for (int j = 0; j < half; j++)
            t[half + j] = b;
        return t;
    }

    // Koch et al. 2015 §3.2: a Siamese verification network maps a PAIR (a, b) to
    // a SYMMETRIC similarity sim(a, b) = sigmoid(Sum_j alpha_j |emb(a)_j - emb(b)_j|).
    // The base DifferentInputs tests feed CreateConstantTensor, which makes both
    // halves of the [1, 2, 128] pair identical (a == b) -> |emb(a) - emb(b)| = 0 ->
    // sim = sigmoid(bias) for EVERY such input. That constant output is the correct
    // behaviour of a symmetric metric (identical items are maximally similar
    // regardless of their value), so the base "distinct constant inputs -> distinct
    // outputs" assumption does not hold for this model. We exercise the real
    // contract instead: two pairs with DIFFERENT inter-item distances must yield
    // different similarity scores. This still catches the collapse bug the base
    // test targets (a network stuck at uniform output fails both pairs).
    public override async Task DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var network = CreateNetwork();

        var input1 = MakePair(0.1f, 0.9f); // large inter-item distance
        var input2 = MakePair(0.3f, 0.4f); // small inter-item distance

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ConvertToDouble(output1[i]) - ConvertToDouble(output2[i])) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Siamese network produced identical similarity for pairs with distinct " +
            "inter-item distances (0.1/0.9 vs 0.3/0.4) - the network has collapsed " +
            "to a constant output regardless of the pair distance.");
    }

    public override async Task DifferentInputs_AfterTraining_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        if (TrainingInvariantsNotApplicable(network)) return;

        var trainInput = CreateRandomTensor(InputShape, rng);
        var trainTarget = CreateRandomTargetTensor(EffectiveOutputShape, rng);
        for (int i = 0; i < TrainingIterations; i++)
            network.Train(trainInput, trainTarget);

        var input1 = MakePair(0.1f, 0.9f);
        var input2 = MakePair(0.3f, 0.4f);

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        double sumSquared = 0;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            double d = ConvertToDouble(output1[i]) - ConvertToDouble(output2[i]);
            sumSquared += d * d;
        }
        double l2Distance = Math.Sqrt(sumSquared);

        Assert.True(l2Distance > 1e-9,
            $"Siamese network produced identical similarity for pairs with distinct " +
            $"inter-item distances AFTER training: L2 = {l2Distance:E3}. The network " +
            $"collapsed to a uniform-output state (broken gradient flow to the shared " +
            $"subnetwork or the L1 similarity head).");
    }
}
