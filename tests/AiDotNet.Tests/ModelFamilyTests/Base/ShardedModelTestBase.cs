using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Parts for sharded-wrapper fixtures: a seeded network to wrap and a world-size-1 configuration (#2139).</summary>
public static class ShardedFixtureParts
{
    /// <summary>The configured learning rate a wrapper steps with.</summary>
    public const double LearningRate = 0.01;

    /// <summary>A small seeded network; every call builds an identical one.</summary>
    public static NeuralNetwork<double> Network()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(6, activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(2, activationFunction: (IActivationFunction<double>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 2, layers: layers)
        { RandomSeed = 7 };
        var network = new NeuralNetwork<double>(architecture);
        _ = network.Predict(new Tensor<double>(new[] { 1, 4 }));
        return network;
    }

    /// <summary>A single-process configuration over its own in-memory backend.</summary>
    public static ShardingConfiguration<double> Config()
    {
        var backend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        backend.Initialize();
        return new ShardingConfiguration<double>(backend, LearningRate);
    }
}

/// <summary>
/// Family invariants for sharded training wrappers (DDP, FSDP, ZeRO-1/2, tensor, pipeline and hybrid parallel) (#2139).
/// </summary>
/// <remarks>
/// At world size 1 every collective is the identity, so a wrapper must be exactly the model it wraps: the same
/// predictions, and a training step equal to one SGD step at the configured learning rate on the same gradients.
/// Writing this family found ZeRO-1 and ZeRO-2 throwing on their first Predict (their shard was never initialized) and
/// ZeRO-1 ignoring the configured learning rate.
/// </remarks>
public abstract class ShardedModelTestBase
{
    /// <summary>Subclasses wrap <see cref="ShardedFixtureParts.Network"/> with <see cref="ShardedFixtureParts.Config"/>.</summary>
    protected abstract IFullModel<double, Tensor<double>, Tensor<double>> CreateWrapper();

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_AtWorldSizeOne_IsTheWrappedModels()
    {
        await Task.Yield();
        var wrapper = CreateWrapper();
        var twin = ShardedFixtureParts.Network();
        var input = Random(new[] { 8, 4 }, 1);

        var expected = twin.Predict(input);
        var actual = wrapper.Predict(input);

        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Train_AtWorldSizeOne_IsOneSgdStepAtTheConfiguredRate()
    {
        await Task.Yield();
        var wrapper = CreateWrapper();
        var twin = ShardedFixtureParts.Network();
        var input = Random(new[] { 8, 4 }, 2);
        var target = Random(new[] { 8, 2 }, 3);
        var before = twin.GetParameters();

        twin.ApplyGradients(twin.ComputeGradients(input, target), ShardedFixtureParts.LearningRate);
        wrapper.Train(input, target);

        var expected = twin.GetParameters();
        var actual = wrapper.GetParameters();
        Assert.Equal(expected.Length, actual.Length);
        double moved = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
            moved = Math.Max(moved, Math.Abs(expected[i] - before[i]));
        }
        Assert.True(moved > 0, "The step changed no parameter, so it cannot tell a wrapper that trains from one that does not.");
    }
}
