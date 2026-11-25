using System;
using AiDotNet.Audio.Enhancement;
using AiDotNet.ComputerVision.Segmentation.OpenVocabulary;
using AiDotNet.ComputerVision.Segmentation.Referring;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Arena;

/// <summary>
/// State a model keeps for its lifetime must not come from an ambient <see cref="TensorArena"/>. The model performance
/// census builds each model inside an arena and resets it between requests, exactly as a serving loop would; a model
/// that took its persistent tensors from the arena then trained against recycled buffers. These are the shapes the
/// census failed on (#2390):
/// <list type="bullet">
/// <item>TabNet's decision/attention selectors, built in the constructor with engine ops (backward: "Cannot reshape
/// tensor with 64 elements to shape [1, 128]").</item>
/// <item>The open-vocabulary segmenters' normalized class-name embeddings, built in the constructor ("Text embeddings
/// must be [K, D]").</item>
/// <item>DeepFilterNet's ERB pooling/expansion matrices and analysis window, cached on the first forward (backward:
/// ArgumentOutOfRangeException in the GEMM input gradient).</item>
/// </list>
/// </summary>
[CollectionDefinition(nameof(ArenaLifetimeModelStateTests), DisableParallelization = true)]
public sealed class ArenaLifetimeModelStateTestsCollection { }

[Collection(nameof(ArenaLifetimeModelStateTests))]
public class ArenaLifetimeModelStateTests
{
    [Fact(Timeout = 120000)]
    public Task TabNet_ConstructedInsideAnArena_TrainsAfterReset()
        => AssertTrainsAfterArenaReset(() => new TabNetNetwork<float>(), new[] { 16 });

    [Fact(Timeout = 120000)]
    public Task OpenVocabularySegmenter_ConstructedInsideAnArena_TrainsAfterReset()
        => AssertTrainsAfterArenaReset(
            () => new SAN<float>(ImageArchitecture(), new DeterministicTextEncoder<float>(), options: new SANOptions { NumClasses = 4 }),
            new[] { 3, 64, 64 });

    [Fact(Timeout = 120000)]
    public Task ReferringSegmenter_ConstructedInsideAnArena_TrainsAfterReset()
        => AssertTrainsAfterArenaReset(
            () => new LISA<double>(new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
                NetworkComplexity.Deep, 0, 32, 32, 3, 0), new DeterministicTextEncoder<double>()), new[] { 1, 3, 32, 32 });

    [Fact(Timeout = 120000)]
    public Task DeepFilterNet_FirstForwardInsideAnArena_TrainsAfterReset()
        => AssertTrainsAfterArenaReset(() => new DeepFilterNet<float>(), new[] { 1, 64, 32 });

    private static NeuralNetworkArchitecture<float> ImageArchitecture()
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.MultiClassClassification, NetworkComplexity.Deep, 0, 64, 64, 3, 4);

    /// <summary>
    /// The census lifecycle: build and run the model inside an arena, reset it, let other work take the recycled
    /// buffers, then build the training objective and its gradients. The precision, input shapes and segmenter sizes are the census fixtures'.
    /// </summary>
    private static async Task AssertTrainsAfterArenaReset<T>(Func<NeuralNetworkBase<T>> create, int[] shape)
    {
        await Task.Yield();
        AiDotNetEngine.ResetToCpu();
        var engine = AiDotNetEngine.Current;
        using var arena = TensorArena.Create();
        using var network = create();
        var input = Filled<T>(shape, 0.25);

        network.SetTrainingMode(false);
        _ = network.Predict(input);
        arena.Reset();

        // Steady requests: each reissues the recycled buffers in the order the first request allocated them.
        for (int request = 0; request < 3; request++)
        {
            _ = network.Predict(input);
            arena.Reset();
        }

        // Reissue every recycled buffer with other contents and shapes, as the next request would.
        for (int size = 1; size <= 1 << 16; size *= 2)
        {
            _ = engine.TensorAdd(Filled<T>(new[] { size }, 1.0), Filled<T>(new[] { size }, 7.0));
            _ = engine.TensorAdd(Filled<T>(new[] { 1, size }, 1.0), Filled<T>(new[] { 1, size }, -3.0));
        }

        Tensor<T> target;
        using (new GradientTape<T>())
        {
            var trainingOutput = network.ForwardPreparedForTraining(input);
            target = Filled<T>(trainingOutput.Shape.ToArray(), 0.5);
        }

        arena.Reset();
        using var tape = new GradientTape<T>();
        var objective = network.BuildTrainingObjective(input, target, network.DefaultLossFunction);
        var gradients = tape.ComputeGradients(objective, sources: null);
        Assert.NotEmpty(gradients);
        Assert.False(double.IsNaN(Convert.ToDouble(objective.ToArray()[0])));
    }

    private static Tensor<T> Filled<T>(int[] shape, double value)
    {
        int length = 1;
        foreach (int dim in shape) length *= dim;
        var ops = MathHelper.GetNumericOperations<T>();
        var data = new T[length];
        for (int i = 0; i < length; i++) data[i] = ops.FromDouble(value * (1.0 + 0.01 * (i % 17)));
        return new Tensor<T>(shape, new Vector<T>(data));
    }
}
