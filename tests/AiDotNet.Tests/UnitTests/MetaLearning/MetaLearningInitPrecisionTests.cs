using System;
using System.Reflection;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Components;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Random initial weights in Half are each rounded ONCE from the double value the draw defines, as they were before the
/// element-wise conversion. Built in Half, u was rounded first and then shifted (u - 0.5, 2u - 1 or 1 + 2s(u - 0.5)), which
/// cancels the leading bits of the draw, and constants such as 0.01 or 1 / sqrt(fan-in) were rounded before they scaled it.
/// </summary>
public class MetaLearningInitPrecisionTests
{
    private static Vector<Half> Field(object owner, string name)
    {
        var field = owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        var value = field.GetValue(owner) as Vector<Half>;
        Assert.NotNull(value);
        return value;
    }

    [Fact]
    public void TaskConditionedHyperNetwork_WeightsAreRoundedOnce()
    {
        var algorithm = new TaskCondHyperNetAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new TaskCondHyperNetOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 40))
            {
                EmbeddingDim = 4, HyperHiddenDim = 8, ChunkSize = 8, Seed = 13,
            });
        var weights = algorithm.HyperNetWeights;

        var rng = RandomHelper.CreateSeededRandom(13);
        double scale = 1.0 / Math.Sqrt(8);
        Assert.Equal(4 * 8 + 8 + 5 * 8 * 8, weights.Length);
        for (int i = 0; i < weights.Length; i++)
            Assert.Equal((Half)((rng.NextDouble() - 0.5) * scale), weights[i]);
    }

    [Fact]
    public void RecurrentHyperNetwork_GruWeightsAreRoundedOnce()
    {
        var algorithm = new RecurrentHyperNetAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new RecurrentHyperNetOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 10))
            {
                HiddenStateDim = 4, InputDim = 4, Seed = 13,
            });
        var weights = Field(algorithm, "_gruWeights");

        var rng = RandomHelper.CreateSeededRandom(13);
        double scale = 1.0 / Math.Sqrt(8);
        Assert.Equal(3 * 8 * 4, weights.Length);
        for (int i = 0; i < weights.Length; i++)
            Assert.Equal((Half)((rng.NextDouble() - 0.5) * scale), weights[i]);
    }

    [Fact]
    public void CompressionLemmaEstimator_FirstLayerIsRoundedOnce()
    {
        var estimator = new CompressionLemmaKLEstimator<Half>(inputDimension: 20, hiddenWidth: 8, rng: RandomHelper.CreateSeededRandom(3));
        var omega = estimator.GetParameters();

        var rng = RandomHelper.CreateSeededRandom(3);
        double scale = Math.Sqrt(1.0 / 20) * 1e-3;
        for (int i = 0; i < 20 * 8; i++)
            Assert.Equal((Half)(((rng.NextDouble() * 2.0) - 1.0) * scale), omega[i]);
    }

    [Fact]
    public void ImplicitPosteriorGenerator_WeightsAreRoundedOnce()
    {
        // Layout W1 [4 x 5], b1 [5], W2 [5 x 3], b2 [3], W3 [3 x 6], b3 [6]; the draws fill W1, W2, W3 in order.
        var generator = new ImplicitPosteriorGenerator<Half>(
            outputDimension: 6, latentDimension: 4, firstHiddenWidth: 5, secondHiddenWidth: 3, rng: RandomHelper.CreateSeededRandom(3));
        var lambda = generator.GetParameters();

        var rng = RandomHelper.CreateSeededRandom(3);
        void Check(int offset, int count, int fanIn)
        {
            double scale = Math.Sqrt(1.0 / fanIn);
            for (int i = 0; i < count; i++)
                Assert.Equal((Half)(((rng.NextDouble() * 2.0) - 1.0) * scale), lambda[offset + i]);
        }

        Check(0, 20, 4);
        Check(20 + 5, 15, 5);
        Check(20 + 5 + 15 + 3, 18, 3);
    }

    [Fact]
    public void WarpGrad_WarpParametersAreRoundedOnce()
    {
        const double s = 0.9;
        var algorithm = new WarpGradAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new WarpGradOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 200))
            {
                NumWarpLayers = 2, UseDiagonalWarp = true, WarpInitScale = s, Seed = 5,
            });

        var rng = RandomHelper.CreateSeededRandom(5);
        Assert.Equal(2, algorithm.WarpLayerParameters.Count);
        foreach (var layer in algorithm.WarpLayerParameters)
        {
            Assert.Equal(100, layer.Length);
            for (int j = 0; j < layer.Length; j++)
                Assert.Equal((Half)(((rng.NextDouble() - 0.5) * (2.0 * s)) + 1.0), layer[j]);
        }
    }

    [Fact]
    public void PEARL_EncoderWeightsAreRoundedOnce()
    {
        var algorithm = new PEARLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new PEARLOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 16))
            {
                LatentDim = 2, EncoderHiddenDim = 16, Seed = 21,
            });
        var encoder = Field(algorithm, "_encoderParams");

        // MetaLearnerBase.SampleNormal: u1 = 1 - U, u2 = U, z = sqrt(-2 ln u1) cos(2 pi u2); the weight is 0.01 z.
        var rng = RandomHelper.CreateSeededRandom(21);
        Assert.Equal(16 * 16 + 16 * 2 * 2, encoder.Length);
        for (int i = 0; i < encoder.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            Assert.Equal((Half)(z * 0.01), encoder[i]);
        }
    }
}
