using System;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// <see cref="NoisePredictorBase{T}.ComputeGradientsWithTape"/> takes its gradients through the shared tape helper:
/// they must be the exact gradients of the default MSE objective, also when the caller trains inside an arena and
/// keeps allocating after the call.
/// </summary>
public class NoisePredictorTapeGradientTests
{
    private const int N = 4096;

    private sealed class FixturePredictor : NoisePredictorBase<double>
    {
        public override int InputChannels => 1;
        public override int OutputChannels => 1;
        public override int BaseChannels => 1;
        public override int TimeEmbeddingDim => 1;
        public override bool SupportsCFG => false;
        public override bool SupportsCrossAttention => false;
        public override int ContextDimension => 0;
        public override Tensor<double> PredictNoise(Tensor<double> noisySample, int timestep, Tensor<double>? conditioning = null)
            => noisySample;
    }

    private static Tensor<double> Wave(double phase)
    {
        var t = new Tensor<double>(new[] { N });
        for (int i = 0; i < N; i++) t[i] = Math.Sin(0.01 * i + phase);
        return t;
    }

    [Fact]
    public void Gradients_AreExact_UnderAnArena()
    {
        var x = Wave(0.0);
        var target = Wave(1.0);
        var w = Wave(2.0);
        var engine = AiDotNetEngine.Current;
        var predictor = new FixturePredictor();

        using var arena = TensorArena.Create();
        var gradients = predictor.ComputeGradientsWithTape(
            x, target, new[] { w }, forwardBuilder: input => engine.TensorMultiply(input, w));

        // Allocations after the tape closed, of the gradient's shape, with values unlike any gradient.
        for (int k = 0; k < 16; k++)
        {
            var filler = engine.TensorAdd(engine.TensorMultiplyScalar(x, 1000.0 + k), w);
            Assert.Equal(N, filler.Length);
        }

        // d/dw mean((x w - t)^2) = 2 / n * (x w - t) * x.
        var gradient = gradients[w];
        for (int i = 0; i < N; i++)
        {
            double expected = 2.0 / N * (x[i] * w[i] - target[i]) * x[i];
            Assert.True(Math.Abs(expected - gradient[i]) <= 1e-15, $"dL/dw[{i}]: {gradient[i]:G17} vs {expected:G17}.");
        }
    }
}