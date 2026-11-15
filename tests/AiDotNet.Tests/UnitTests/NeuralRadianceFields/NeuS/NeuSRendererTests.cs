using System;
using System.Linq;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields.NeuS;

/// <summary>
/// The NeuS renderer's maths (Wang et al. 2021) against hand-derived answers: the discrete opacity, the weights, their
/// unbiasedness at a plane's zero crossing, the S-density sharpness, and the host sampling helpers.
/// </summary>
public class NeuSRendererTests
{
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    private static Tensor<double> T(double[] values, params int[] shape) => NeuSHost<double>.FromDoubles(values, shape);

    private static double[] H(Tensor<double> tensor) => NeuSHost<double>.ToDoubles(tensor);

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    [Fact]
    public void InverseS_IsExpOfTenTimesVariance()
    {
        var s = H(NeuSRenderer<double>.InverseS(Engine, T(new[] { 0.3 }, 1)));
        Assert.Single(s);
        Assert.Equal(Math.Exp(3.0), s[0], 9);
    }

    [Fact]
    public void Alpha_MatchesHandReference()
    {
        var sdf = new[] { 0.4, 0.05, -0.1, -0.3 };
        var cos = new[] { -0.9, 0.2, -1.0, -0.4 };
        var dist = new[] { 0.1, 0.2, 0.05, 0.3 };
        const double s = 12.0, ratio = 0.3;
        var alpha = H(NeuSRenderer<double>.Alpha(Engine, T(sdf, 4, 1), T(cos, 4, 1), T(dist, 4, 1), T(new[] { s }, 1, 1),
            ratio, null, null));
        for (int i = 0; i < 4; i++)
        {
            // render_core: iter_cos = −(relu(−cos/2 + 0.5)(1 − r) + relu(−cos) r).
            double iterCos = -(Math.Max(0, -cos[i] * 0.5 + 0.5) * (1 - ratio) + Math.Max(0, -cos[i]) * ratio);
            double prev = Sigmoid((sdf[i] - iterCos * dist[i] * 0.5) * s);
            double next = Sigmoid((sdf[i] + iterCos * dist[i] * 0.5) * s);
            double expected = Math.Min(1, Math.Max(0, (prev - next + 1e-5) / (prev + 1e-5)));
            Assert.Equal(expected, alpha[i], 12);
        }
    }

    [Fact]
    public void Alpha_ClipsTheSlopeAndAppliesThePointMask()
    {
        // SparseNeuS: iter_cos clipped to ±10, masked samples are transparent.
        var alpha = H(NeuSRenderer<double>.Alpha(Engine, T(new[] { 0.0, 0.0 }, 2, 1), T(new[] { -50.0, -50.0 }, 2, 1),
            T(new[] { 0.1, 0.1 }, 2, 1), T(new[] { 4.0 }, 1, 1), 1.0, 10.0, T(new[] { 1.0, 0.0 }, 2, 1)));
        double prev = Sigmoid(10 * 0.1 * 0.5 * 4), next = Sigmoid(-10 * 0.1 * 0.5 * 4);
        Assert.Equal((prev - next + 1e-5) / (prev + 1e-5), alpha[0], 12);
        Assert.Equal(0.0, alpha[1], 12);
    }

    [Fact]
    public void Weights_AreAlphaTimesExclusiveTransmittance()
    {
        var alpha = new[] { 0.2, 0.5, 0.1, 0.0, 0.9, 0.3 };
        var w = H(NeuSRenderer<double>.Weights(Engine, T(alpha, 2, 3)));
        const double e = 1e-7;
        Assert.Equal(0.2, w[0], 12);
        Assert.Equal(0.5 * (0.8 + e), w[1], 12);
        Assert.Equal(0.1 * (0.8 + e) * (0.5 + e), w[2], 12);
        Assert.Equal(0.0, w[3], 12);
        Assert.Equal(0.9 * (1.0 + e), w[4], 12);
        Assert.Equal(0.3 * (1.0 + e) * (0.1 + e), w[5], 12);
    }

    [Theory]
    [InlineData(0.3)]
    [InlineData(0.4)]
    public void Weights_PeakAtAPlanesZeroCrossing_AndAreUnbiased(double variance)
    {
        // A ray along +z through the plane f(p) = 1 − z: the surface is at depth 1, ∇f·d = −1.
        const int samples = 21;
        var z = Enumerable.Range(0, samples).Select(i => i * 0.1).ToArray();
        var (dists, mid) = NeuSRenderer<double>.Sections(z, samples, 0.1);
        var sdf = mid.Select(m => 1.0 - m).ToArray();
        var alpha = NeuSRenderer<double>.Alpha(Engine, T(sdf, samples, 1), T(Enumerable.Repeat(-1.0, samples).ToArray(), samples, 1),
            T(dists, samples, 1), NeuSRenderer<double>.InverseS(Engine, T(new[] { variance }, 1)), 1.0, null, null);
        var w = H(NeuSRenderer<double>.Weights(Engine, Engine.Reshape(alpha, new[] { 1, samples })));

        int peak = Array.IndexOf(w, w.Max());
        Assert.True(z[peak] <= 1.0 + 1e-9 && z[peak] + dists[peak] >= 1.0 - 1e-9,
            $"The heaviest section [{z[peak]}, {z[peak] + dists[peak]}] should contain the surface at depth 1.");
        // wᵢ = Φ(fᵢ) − Φ(fᵢ₊₁) over Φ(f₀) ≈ 1: symmetric about the crossing, so the expected depth is the surface.
        double sum = w.Sum();
        double depth = w.Select((wi, i) => wi * mid[i]).Sum() / sum;
        Assert.Equal(1.0, depth, 3);
        Assert.InRange(sum, 0.99, 1.0 + 1e-4); // the 1e-7 transmittance pad can lift the sum by S·1e-7
        Assert.True(w[peak - 2] < w[peak - 1] && w[peak - 1] <= w[peak], "Weights rise towards the crossing.");
        Assert.True(w[peak + 2] < w[peak + 1] && w[peak + 1] <= w[peak], "Weights fall after the crossing.");
    }

    [Fact]
    public void Composite_IsTheWeightedSumPerRay()
    {
        var weights = T(new[] { 0.5, 0.25, 1.0, 0.0 }, 2, 2);
        var values = T(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 }, 4, 2);
        var c = H(NeuSRenderer<double>.Composite(Engine, weights, values));
        Assert.Equal(new[] { 0.5 * 1 + 0.25 * 3, 0.5 * 2 + 0.25 * 4, 5.0, 6.0 }, c);
    }

    [Fact]
    public void EikonalError_IsTheMaskedMeanSquaredNormResidual()
    {
        var g = T(new[] { 3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0 }, 3, 3);
        var e = H(NeuSRenderer<double>.EikonalError(Engine, g, new[] { 1.0, 1.0, 0.0 }));
        Assert.Equal((16.0 + 0.0) / (2.0 + 1e-5), e[0], 10);
    }

    [Fact]
    public void NearFarFromSphere_IsTheUnitSphereChordAroundTheClosestPoint()
    {
        var (near, far) = NeuSSampling.NearFarFromSphere(new[] { 0.0, 0.0, -3.0 }, new[] { 0.0, 0.0, 1.0 });
        Assert.Equal(2.0, near[0], 12);
        Assert.Equal(4.0, far[0], 12);
    }

    [Fact]
    public void SamplePdf_UniformWeightsGiveEvenQuantiles()
    {
        var bins = new[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var weights = new[] { 1.0, 1.0, 1.0, 1.0 };
        var z = NeuSSampling.SamplePdf(bins, weights, 5, 4, null);
        // Quantiles (k + 0.5)/4 of a uniform density over [0, 4].
        Assert.Equal(new[] { 0.5, 1.5, 2.5, 3.5 }, z.Select(v => Math.Round(v, 9)).ToArray());
    }

    [Fact]
    public void SamplePdf_ConcentratesInTheHeavySection()
    {
        var bins = new[] { 0.0, 1.0, 2.0, 3.0 };
        var z = NeuSSampling.SamplePdf(bins, new[] { 0.0, 1000.0, 0.0 }, 4, 8, null);
        Assert.All(z, v => Assert.InRange(v, 1.0, 2.0));
        for (int i = 1; i < z.Length; i++) Assert.True(z[i] >= z[i - 1]);
    }

    [Fact]
    public void Merge_SortsDepthsAndCarriesValues()
    {
        var (z, values) = NeuSSampling.Merge(new[] { 0.0, 1.0, 2.0 }, new[] { 10.0, 11.0, 12.0 }, 3,
            new[] { 1.5, 0.5 }, new[] { 21.0, 20.0 }, 2);
        Assert.Equal(new[] { 0.0, 0.5, 1.0, 1.5, 2.0 }, z);
        Assert.NotNull(values);
        Assert.Equal(new[] { 10.0, 20.0, 11.0, 21.0, 12.0 }, values);
    }

    [Fact]
    public void UpSample_PlacesNewSamplesNearTheZeroCrossing()
    {
        const int samples = 33;
        var origins = new[] { 0.0, 0.0, -2.0 };
        var directions = new[] { 0.0, 0.0, 1.0 };
        var (near, far) = NeuSSampling.NearFarFromSphere(origins, directions);
        var z = NeuSSampling.UniformDepths(near, far, samples, null);
        // Sphere of radius 0.5 at the origin: the ray enters it at depth 1.5.
        var sdf = z.Select(d => Math.Abs(-2.0 + d) - 0.5).ToArray();
        var newZ = NeuSSampling.UpSample(origins, directions, z, sdf, samples, 16, 64.0, -1000.0, null);
        Assert.Equal(16, newZ.Length);
        double meanDepth = newZ.Average();
        Assert.InRange(meanDepth, 1.4, 1.6);
    }

    [Fact]
    public void Embedder_OrdersInputThenSinesThenCosinesPerFrequency()
    {
        var embedder = new NeuSFrequencyEmbedder<double>(2, 2);
        Assert.Equal(10, embedder.OutputDims);
        var x = new[] { 0.3, -0.2 };
        var e = H(embedder.Embed(Engine, T(x, 1, 2)));
        var expected = new[]
        {
            0.3, -0.2,
            Math.Sin(0.3), Math.Sin(-0.2), Math.Cos(0.3), Math.Cos(-0.2),
            Math.Sin(0.6), Math.Sin(-0.4), Math.Cos(0.6), Math.Cos(-0.4),
        };
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], e[i], 12);
    }

    [Fact]
    public void Embedder_BackwardIsTheVectorJacobianProduct()
    {
        var embedder = new NeuSFrequencyEmbedder<double>(3, 3);
        var random = RandomHelper.CreateSeededRandom(5);
        var x = Enumerable.Range(0, 6).Select(_ => random.NextDouble() * 2 - 1).ToArray();
        var g = Enumerable.Range(0, 2 * embedder.OutputDims).Select(_ => random.NextDouble() * 2 - 1).ToArray();
        var analytic = H(embedder.Backward(Engine, T(x, 2, 3), T(g, 2, embedder.OutputDims)));
        const double h = 1e-6;
        for (int i = 0; i < x.Length; i++)
        {
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[i] += h;
            minus[i] -= h;
            var ep = H(embedder.Embed(Engine, T(plus, 2, 3)));
            var em = H(embedder.Embed(Engine, T(minus, 2, 3)));
            double numeric = 0;
            for (int k = 0; k < ep.Length; k++) numeric += g[k] * (ep[k] - em[k]) / (2 * h);
            Assert.Equal(numeric, analytic[i], 6);
        }
    }
}
