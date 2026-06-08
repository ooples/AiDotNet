using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Guards the cross-test / cross-process weight-initialization determinism fix:
/// when a layer has a pinned <see cref="LayerBase{T}.RandomSeed"/>, its weight
/// initialization must derive from that seed and be INDEPENDENT of the shared
/// <see cref="RandomHelper.ThreadSafeRandom"/>'s cumulative state. The regression
/// being guarded: a DenseLayer constructed with a LAZY initialization strategy
/// (e.g. the VITS input-projection layer) ran a generic, non-seeded Xavier fill
/// instead of its seeded init, so its initial weights depended on how much
/// unrelated work had already advanced the shared RNG — an invariant that held in
/// isolation flipped once other work ran first.
/// </summary>
public class InitializationDeterminismTests
{
    private static double[] BuildAndInitDenseLazyStrategy(int seed, int inputSize, int outputSize)
    {
        // Lazy strategy is the exact case that bypassed the seeded path.
        var layer = new DenseLayer<double>(
            outputSize,
            new IdentityActivation<double>() as IActivationFunction<double>,
            InitializationStrategies<double>.Lazy)
        {
            RandomSeed = seed,
        };
        // First forward resolves the lazy input shape and triggers initialization.
        layer.Forward(new Tensor<double>(new[] { 1, inputSize }));
        var p = layer.GetParameters();
        var arr = new double[p.Length];
        for (int i = 0; i < p.Length; i++) arr[i] = p[i];
        return arr;
    }

    [Fact]
    public void DenseLayer_LazyStrategy_WithRandomSeed_InitIsDeterministic_DespiteSharedRngDrain()
    {
        double[] first = BuildAndInitDenseLazyStrategy(seed: 4242, inputSize: 8, outputSize: 16);

        // Advance the shared RNG by an arbitrary amount, simulating the cumulative
        // draws that prior tests / prior work perform in the same process.
        for (int i = 0; i < 5000; i++) RandomHelper.ThreadSafeRandom.Next();

        double[] second = BuildAndInitDenseLazyStrategy(seed: 4242, inputSize: 8, outputSize: 16);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i], 12);
    }

    private static double[] BuildAndInitConv1DHe(int seed, int inputChannels, int outputChannels, int kernelSize)
    {
        // Conv1DLayer defaults to the He strategy, which is NON-LAZY — the exact
        // case that bypassed the RandomSeed-derived path (InitializeLayerWeights
        // early-returned for any non-lazy strategy, letting He sample from the
        // process-shared ThreadSafeRandom). Use the lazy-input-channel constructor
        // so initialization runs at first Forward, AFTER RandomSeed is assigned.
        var layer = new Conv1DLayer<double>(outputChannels: outputChannels, kernelSize: kernelSize)
        {
            RandomSeed = seed,
        };
        layer.Forward(new Tensor<double>(new[] { 1, inputChannels, 8 }));
        var p = layer.GetParameters();
        var arr = new double[p.Length];
        for (int i = 0; i < p.Length; i++) arr[i] = p[i];
        return arr;
    }

    [Fact]
    public void Conv1DLayer_HeStrategy_WithRandomSeed_InitIsDeterministic_DespiteSharedRngDrain()
    {
        double[] first = BuildAndInitConv1DHe(seed: 4242, inputChannels: 8, outputChannels: 16, kernelSize: 3);

        // Advance the shared RNG to simulate prior in-process work. Before the fix
        // this changed He's draws and the conv weights diverged; after it the
        // seeded He path is independent of the shared RNG.
        for (int i = 0; i < 5000; i++) RandomHelper.ThreadSafeRandom.Next();

        double[] second = BuildAndInitConv1DHe(seed: 4242, inputChannels: 8, outputChannels: 16, kernelSize: 3);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i], 12);
    }

    [Fact]
    public void NeuralNetwork_SeededArchitecture_InitIsDeterministic_DespiteSharedRngDrain()
    {
        static double[] InitParams()
        {
            var arch = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 16,
                outputSize: 4,
                complexity: NetworkComplexity.Simple)
            {
                RandomSeed = 1234,
            };
            using var net = new FeedForwardNeuralNetwork<double>(arch);
            var p = net.GetParameters();
            var arr = new double[p.Length];
            for (int i = 0; i < p.Length; i++) arr[i] = p[i];
            return arr;
        }

        double[] a = InitParams();
        for (int i = 0; i < 5000; i++) RandomHelper.ThreadSafeRandom.Next();
        double[] b = InitParams();

        // Same architecture seed → identical initial weights regardless of how the
        // shared RNG was advanced between constructions. Compare the FULL parameter
        // vectors, not an aggregate sum: a sum can collide across different weight
        // vectors (e.g. a permutation or compensating sign flips), which would let a
        // real determinism regression slip through.
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i], 12);
    }
}
