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
/// Correctness guard for the float Predict → CompiledMlp fast path
/// (FeedForwardNeuralNetwork.TryFusedDensePredict). The first Predict on a fresh
/// network runs the generic per-layer Forward (weights are still lazy, so the fused
/// path declines); subsequent Predicts run the CompiledMlp plan. Both must produce
/// the same output to floating-point rounding — i.e. wiring CompiledMlp into Predict
/// does not change results, only speed.
/// </summary>
public class FeedForwardCompiledMlpPredictTests
{
    private static FeedForwardNeuralNetwork<float> BuildMlp(int inSize, int h, int outSize)
    {
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new DenseLayer<float>(h, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(outSize, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inSize,
            outputSize: outSize,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(32)]
    public void Predict_CompiledMlpPath_MatchesGenericForward(int batch)
    {
        const int inSize = 6, h = 5, outSize = 3;
        var model = BuildMlp(inSize, h, outSize);

        var rng = new Random(123);
        var inData = new float[batch * inSize];
        for (int i = 0; i < inData.Length; i++) inData[i] = (float)(rng.NextDouble() * 2 - 1);
        var input = new Tensor<float>(inData, new[] { batch, inSize });

        // First call: weights are lazy → fused path declines → generic per-layer Forward.
        var generic = model.Predict(input).ToArray();
        // Second call: weights materialized → CompiledMlp fast path engages.
        var compiled = model.Predict(input).ToArray();

        Assert.Equal(generic.Length, compiled.Length);
        double maxDiff = 0;
        for (int i = 0; i < generic.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(generic[i] - compiled[i]));
        Assert.True(maxDiff <= 1e-4, $"CompiledMlp Predict diverged from generic Forward by {maxDiff:E3}");

        // Determinism: a third call (also CompiledMlp) must equal the second exactly.
        var compiled2 = model.Predict(input).ToArray();
        for (int i = 0; i < compiled.Length; i++)
            Assert.Equal(compiled[i], compiled2[i]);
    }
}
