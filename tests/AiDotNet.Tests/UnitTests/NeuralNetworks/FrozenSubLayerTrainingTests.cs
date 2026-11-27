using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A sub-layer frozen with <c>FreezeSubLayerParameters</c> (PyTorch's <c>requires_grad = False</c>: a LoRA adapter's
/// base layer) is out of the parameter surface, so a training step must leave its weights unchanged and the trainable
/// count must not include them.
/// </summary>
public sealed class FrozenSubLayerTrainingTests
{
    private const int Features = 6;

    private static (NeuralNetwork<double> Network, DenseLayer<double> Base, StandardLoRAAdapter<double> Adapter) Build()
    {
        var baseLayer = new DenseLayer<double>(Features);
        baseLayer.ResolveFromShape(new[] { Features });
        var adapter = new StandardLoRAAdapter<double>(baseLayer, rank: 2, alpha: 2, freezeBaseLayer: true);
        var head = new DenseLayer<double>(Features);
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: Features,
            outputSize: Features,
            layers: new List<ILayer<double>> { adapter, head }) { RandomSeed = 7 };
        return (new NeuralNetwork<double>(architecture), baseLayer, adapter);
    }

    private static Tensor<double> Filled(int rows, int salt)
    {
        var t = new Tensor<double>(new[] { rows, Features });
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, Math.Sin(0.37 * i + salt));
        return t;
    }

    [Fact]
    public void A_training_step_leaves_a_frozen_sub_layer_unchanged()
    {
        var (network, baseLayer, adapter) = Build();
        network.SetTrainingMode(true);
        var x = Filled(4, 1);
        var y = Filled(4, 2);
        network.Predict(x);   // materializes lazy weights
        var baseBefore = baseLayer.GetParameters().ToArray();
        var adapterBefore = adapter.GetParameters().ToArray();
        for (int step = 0; step < 3; step++) network.Train(x, y);
        Assert.Equal(baseBefore, baseLayer.GetParameters().ToArray());
        // The step did train: the adapter's own (LoRA) parameters moved.
        Assert.NotEqual(adapterBefore, adapter.GetParameters().ToArray());
    }

    [Fact]
    public void The_trainable_count_excludes_a_frozen_sub_layer()
    {
        var (network, baseLayer, adapter) = Build();
        network.Predict(Filled(1, 3));
        Assert.Equal(adapter.ParameterCount, adapter.TrainableParameterCount);
        Assert.True(adapter.TrainableParameterCount < adapter.ParameterCount + baseLayer.ParameterCount);
        Assert.Equal(network.ParameterCount, network.TrainableParameterCount);
    }
}
