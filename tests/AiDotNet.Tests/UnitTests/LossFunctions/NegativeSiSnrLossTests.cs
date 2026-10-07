using System;
using System.Linq;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

/// <summary>Negative SI-SNR with permutation-invariant training (Luo &amp; Mesgarani 2019, Sec. III-D).</summary>
public class NegativeSiSnrLossTests
{
    private static Tensor<double> Random(int seed, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() - 0.5;
        return tensor;
    }

    [Fact]
    public void Loss_IsScaleInvariant()
    {
        var loss = new NegativeSiSnrLoss<double>();
        var estimate = Random(1, 1, 1, 64);
        var reference = Random(2, 1, 1, 64);
        var scaled = new Tensor<double>(estimate.Shape.ToArray());
        for (int i = 0; i < estimate.Length; i++) scaled[i] = 7.5 * estimate[i];

        // Exact only without the 1e-8 stabiliser added to each energy (as every reference implementation does);
        // rescaling moves the energies against that fixed term, here by about 1.5e-6 dB, far below anything a
        // training signal could act on.
        Assert.Equal(loss.ComputeTapeLoss(estimate, reference)[0], loss.ComputeTapeLoss(scaled, reference)[0], 4);
    }

    [Fact]
    public void Loss_IsInvariantToTheOrderOfTheEstimatedSources()
    {
        var loss = new NegativeSiSnrLoss<double>();
        var reference = Random(3, 2, 2, 48);
        // Near-perfect estimates, then the same estimates with the two sources swapped.
        var estimate = new Tensor<double>(reference.Shape.ToArray());
        var swapped = new Tensor<double>(reference.Shape.ToArray());
        var noise = Random(4, 2, 2, 48);
        for (int b = 0; b < 2; b++)
        for (int s = 0; s < 2; s++)
        for (int t = 0; t < 48; t++)
        {
            double value = reference[b, s, t] + 0.05 * noise[b, s, t];
            estimate[b, s, t] = value;
            swapped[b, 1 - s, t] = value;
        }

        Assert.Equal(loss.ComputeTapeLoss(estimate, reference)[0], loss.ComputeTapeLoss(swapped, reference)[0], 10);
        Assert.True(loss.ComputeTapeLoss(estimate, reference)[0] < -20.0, "A close estimate should score well above 20 dB.");
    }

    [Fact]
    public void TapeLoss_MatchesTheVectorForm_ForOneSource()
    {
        var loss = new NegativeSiSnrLoss<double>();
        var estimate = Random(5, 1, 1, 40);
        var reference = Random(6, 1, 1, 40);

        double tape = loss.ComputeTapeLoss(estimate, reference)[0];
        double vector = loss.CalculateLoss(new Vector<double>(estimate.ToArray()), new Vector<double>(reference.ToArray()));

        Assert.Equal(vector, tape, 8);
    }

    [Fact]
    public void Gradient_MatchesFiniteDifferences()
    {
        var loss = new NegativeSiSnrLoss<double>();
        var estimate = Random(7, 1, 2, 16);
        var reference = Random(8, 1, 2, 16);

        Tensor<double> analytic;
        using (var tape = new GradientTape<double>())
        {
            var value = loss.ComputeTapeLoss(estimate, reference);
            analytic = tape.ComputeGradients(value, new[] { estimate })[estimate];
        }

        const double h = 1e-6;
        for (int i = 0; i < estimate.Length; i++)
        {
            double original = estimate[i];
            estimate[i] = original + h;
            double plus = loss.ComputeTapeLoss(estimate, reference)[0];
            estimate[i] = original - h;
            double minus = loss.ComputeTapeLoss(estimate, reference)[0];
            estimate[i] = original;
            Assert.Equal((plus - minus) / (2 * h), analytic[i], 5);
        }
    }
}