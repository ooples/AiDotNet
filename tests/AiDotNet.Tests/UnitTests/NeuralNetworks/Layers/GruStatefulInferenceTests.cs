using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// A stateful GRU carries its final hidden state into the next Forward call, in inference as in training. Fed a
/// sequence in two halves, it must end where it ends on the whole sequence, on the fused path (the standard
/// activations) and on the per-step path (a candidate activation the fused op does not compute).
/// </summary>
public class GruStatefulInferenceTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TwoHalves_EndWhereTheWholeSequenceEnds(bool perStepPath)
    {
        var layer = new GRULayer<float>(5, returnSequences: false,
            activation: perStepPath ? new ReLUActivation<float>() : (IActivationFunction<float>?)null, stateful: true);
        layer.SetTrainingMode(false);

        var rng = new Random(7);
        var whole = new Tensor<float>(new[] { 2, 6, 3 });
        for (int i = 0; i < whole.Length; i++) whole[i] = (float)(rng.NextDouble() * 2 - 1);
        var first = new Tensor<float>(new[] { 2, 3, 3 });
        var second = new Tensor<float>(new[] { 2, 3, 3 });
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < 3; t++)
                for (int f = 0; f < 3; f++)
                {
                    first[(b * 3 + t) * 3 + f] = whole[(b * 6 + t) * 3 + f];
                    second[(b * 3 + t) * 3 + f] = whole[(b * 6 + t + 3) * 3 + f];
                }

        var expected = layer.Forward(whole).ToArray();
        layer.ResetState();
        layer.Forward(first);
        var actual = layer.Forward(second).ToArray();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f,
                $"[{i}]: whole sequence {expected[i]}, two halves {actual[i]} (the second half did not start from the first's state)");
    }
}
