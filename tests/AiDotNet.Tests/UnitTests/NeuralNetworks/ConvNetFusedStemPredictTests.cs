using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Correctness guard for the fused conv-stem Predict fast path
/// (ConvolutionalNeuralNetwork.TryFusedConvStemPredict). On a fresh network the
/// first Predict runs the generic per-layer Forward (conv weights still lazy, so the
/// fused stem declines); later Predicts run the fused stem. Both must agree to
/// floating-point rounding — the fast path changes speed, not results.
/// </summary>
public class ConvNetFusedStemPredictTests
{
    private static ConvolutionalNeuralNetwork<float> BuildCnn()
    {
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new ConvolutionalLayer<float>(outputDepth: 16, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(4, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 16, inputWidth: 16, inputDepth: 1,
            outputSize: 4,
            layers: layers);
        return new ConvolutionalNeuralNetwork<float>(arch);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void Predict_FusedConvStem_MatchesGenericForward(int batch)
    {
        var model = BuildCnn();
        var rng = new Random(7);
        var data = new float[batch * 1 * 16 * 16];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        var input = new Tensor<float>(data, new[] { batch, 1, 16, 16 });

        var generic = model.Predict(input).ToArray();    // first call: lazy weights → generic Forward
        var fused = model.Predict(input).ToArray();       // weights materialized → fused conv-stem

        Assert.Equal(generic.Length, fused.Length);
        double maxDiff = 0;
        for (int i = 0; i < generic.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(generic[i] - fused[i]));
        Assert.True(maxDiff <= 1e-4, $"Fused conv-stem Predict diverged from generic Forward by {maxDiff:E3}");

        var fused2 = model.Predict(input).ToArray();      // determinism across fused calls
        for (int i = 0; i < fused.Length; i++) Assert.Equal(fused[i], fused2[i]);
    }
}
