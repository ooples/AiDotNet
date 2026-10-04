using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphAttentionNetworkTests : GraphNNModelTestBase<float>
{
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    // The architecture is seeded explicitly so every network starts from the same weights without relying on a
    // static copy of the first network's parameters, which made each test depend on which test built a network
    // first. #1860 was training that differed run to run from identical weights; seeded training is now
    // bit-identical within and across processes, and TrainedParameters_AreBitIdentical_UnderAFixedSeed holds it
    // there.
    private const int Seed = 42;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // dropoutRate: 0 for the invariant suite. The paper default (0.6, Veličković et al. 2018 §3.3)
        // is a TRAINING-TIME regularizer that zeros 60 % of the attention coefficients each step to
        // curb overfitting on real graphs — it is NOT part of the attention mechanism's math. The
        // memorization/gradient probes here test CAPACITY (can the model fit a fixed pair?), the exact
        // opposite of what dropout aids: at 0.6 the per-step gradient is so noisy the loss cannot
        // descend (observed step-1 0.355 → step-100 0.609). Disabling dropout leaves the paper-faithful
        // multi-head attention forward/backward fully exercised while making the capacity probes valid.
        var network = new GraphAttentionNetwork<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.MultiClassClassification,
                inputSize: 128,
                outputSize: 7) { RandomSeed = Seed },
            options: new GraphAttentionNetworkOptions { DropoutRate = 0.0 });
        return network;
    }

    [Fact]
    public void TrainedParameters_AreBitIdentical_UnderAFixedSeed()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        var input = new Tensor<float>(InputShape);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var target = new Tensor<float>(OutputShape);
        for (int row = 0; row < OutputShape[0]; row++) target[row * OutputShape[1] + (row % OutputShape[1])] = 1f;

        float[] Train(out float[] initial)
        {
            var network = CreateNetwork();
            initial = network.GetParameters().ToArray();
            for (int step = 0; step < 20; step++) network.Train(input, target);
            return network.GetParameters().ToArray();
        }

        float[] first = Train(out float[] firstInitial);
        float[] second = Train(out float[] secondInitial);

        Assert.NotEmpty(firstInitial);
        Assert.Equal(firstInitial.Length, first.Length);
        Assert.Equal(first.Length, second.Length);
        Assert.All(first, value => Assert.False(float.IsNaN(value) || float.IsInfinity(value), "a trained parameter is not finite"));
        // Bit patterns, not float equality: float equality treats NaN as unequal to itself and -0f as equal to 0f,
        // so it can pass on vectors that are not identical or fail on ones that are.
        Assert.Equal(Bits(firstInitial), Bits(secondInitial));
        Assert.NotEqual(Bits(firstInitial), Bits(first));
        Assert.Equal(Bits(first), Bits(second));
    }

    private static int[] Bits(float[] values)
    {
        var bits = new int[values.Length];
        for (int i = 0; i < values.Length; i++) bits[i] = BitConverter.ToInt32(BitConverter.GetBytes(values[i]), 0);
        return bits;
    }
}
