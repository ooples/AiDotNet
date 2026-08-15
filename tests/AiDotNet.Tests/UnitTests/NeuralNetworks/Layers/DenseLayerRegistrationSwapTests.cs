using System;
using System.Linq;
using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Establishes what actually happens to a DenseLayer's trainable surface and its engine
/// registration when the weight tensor is REPLACED by a resize.
/// </summary>
public class DenseLayerRegistrationSwapTests
{
    private readonly ITestOutputHelper _out;
    public DenseLayerRegistrationSwapTests(ITestOutputHelper o) => _out = o;

    private static DenseLayer<double> MakeLayer(int outputSize = 3)
        => new(outputSize, (IActivationFunction<double>)new ReLUActivation<double>());

    /// <summary>Reads the private _registeredTensors list that backs engine persistence.</summary>
    private static Tensor<double>[] RegisteredTensors(DenseLayer<double> layer)
    {
        var f = typeof(LayerBase<double>).GetField("_registeredTensors",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        return ((System.Collections.Generic.List<Tensor<double>>)f.GetValue(layer)!).ToArray();
    }

    [Fact]
    public void AfterResize_TrainableSurfaceAndEngineRegistrationBothTrackTheLiveTensor()
    {
        var layer = MakeLayer();
        layer.Forward(new Tensor<double>([2, 4]));

        var beforeTrainable = layer.GetTrainableParameters().First(p => p.Shape.Length == 2);
        var beforeRegistered = RegisteredTensors(layer).First(p => p.Shape.Length == 2);
        _out.WriteLine($"pre-resize : trainable[{beforeTrainable.Shape[0]},{beforeTrainable.Shape[1]}] " +
                       $"registered[{beforeRegistered.Shape[0]},{beforeRegistered.Shape[1]}] " +
                       $"same={ReferenceEquals(beforeTrainable, beforeRegistered)}");

        // Widen the input: DenseLayer.EnsureWeightShapeForInput rebuilds the weight tensor.
        layer.Forward(new Tensor<double>([2, 7]));

        var afterTrainable = layer.GetTrainableParameters().First(p => p.Shape.Length == 2);
        var afterRegistered = RegisteredTensors(layer).First(p => p.Shape.Length == 2);
        _out.WriteLine($"post-resize: trainable[{afterTrainable.Shape[0]},{afterTrainable.Shape[1]}] " +
                       $"registered[{afterRegistered.Shape[0]},{afterRegistered.Shape[1]}] " +
                       $"same={ReferenceEquals(afterTrainable, afterRegistered)}");
        _out.WriteLine($"registered still == pre-resize tensor? {ReferenceEquals(afterRegistered, beforeTrainable)}");

        // The trainable surface is what the tape training step collects and the optimizer writes to.
        Assert.Equal(7, afterTrainable.Shape[0]);

        // The engine registration must track the same tensor; otherwise a GPU engine keeps a
        // persistent handle on a tensor no forward reads, and dispose unregisters the wrong one.
        Assert.True(ReferenceEquals(afterTrainable, afterRegistered),
            $"engine registration points at a [{afterRegistered.Shape[0]},{afterRegistered.Shape[1]}] " +
            $"tensor while the live weights are [{afterTrainable.Shape[0]},{afterTrainable.Shape[1]}].");
    }

    /// <summary>
    /// The registration must keep its ORDER across a resize, not just its contents.
    /// </summary>
    /// <remarks>
    /// SetTrainableParameters pairs the incoming list with _registeredTensors by index against the
    /// order GetTrainableParameters() returns — (weights, biases) for DenseLayer. Fixing the stale
    /// registration by unregister-then-register would append the weights AFTER the biases, so the
    /// next copy-on-write clone would assign the source's weights into this layer's biases. That
    /// swap is shape-compatible only by accident and would corrupt silently.
    /// </remarks>
    [Fact]
    public void AfterResize_RegistrationOrderStillMatchesTrainableOrder()
    {
        var layer = MakeLayer(outputSize: 3);
        layer.Forward(new Tensor<double>([2, 4]));
        layer.Forward(new Tensor<double>([2, 7]));   // triggers the resize

        var trainable = layer.GetTrainableParameters();
        var registered = RegisteredTensors(layer);

        _out.WriteLine("trainable : " + string.Join(", ", trainable.Select(t => $"[{string.Join(",", t.Shape)}]")));
        _out.WriteLine("registered: " + string.Join(", ", registered.Select(t => $"[{string.Join(",", t.Shape)}]")));

        Assert.Equal(trainable.Count, registered.Length);
        for (int i = 0; i < trainable.Count; i++)
            Assert.True(ReferenceEquals(trainable[i], registered[i]),
                $"position {i} differs: GetTrainableParameters() has " +
                $"[{string.Join(",", trainable[i].Shape)}] but the registration has " +
                $"[{string.Join(",", registered[i].Shape)}]. SetTrainableParameters pairs these by " +
                "index, so a clone would receive them transposed.");
    }

    /// <summary>Repeated width changes must not leave dead tensors registered.</summary>
    [Fact]
    public void RepeatedResizes_DoNotAccumulateRegistrations()
    {
        var layer = MakeLayer(outputSize: 3);
        layer.Forward(new Tensor<double>([2, 4]));
        foreach (int width in new[] { 6, 9, 5, 11 })
            layer.Forward(new Tensor<double>([2, width]));

        var registered = RegisteredTensors(layer);
        _out.WriteLine("registered after 4 resizes: " +
            string.Join(", ", registered.Select(t => $"[{string.Join(",", t.Shape)}]")));

        Assert.Equal(2, registered.Length);   // exactly one weights + one biases
        Assert.Equal(11, registered.First(t => t.Shape.Length == 2).Shape[0]);
    }
}
