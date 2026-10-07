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
    public void TapeLoss_EqualsTheBestPermutation_ScoredIndependently()
    {
        // The oracle scores every permutation with the single-signal vector form, which shares no code with the
        // pairwise matrix or the assignment solver.
        const int Batch = 2, Sources = 4, Samples = 32;
        var loss = new NegativeSiSnrLoss<double>();
        var estimate = Random(11, Batch, Sources, Samples);
        var reference = Random(12, Batch, Sources, Samples);

        double expected = 0;
        for (int b = 0; b < Batch; b++)
        {
            double best = double.PositiveInfinity;
            foreach (var order in Orders(Sources))
            {
                double total = 0;
                for (int i = 0; i < Sources; i++)
                    total += loss.CalculateLoss(Row(estimate, b, i, Samples), Row(reference, b, order[i], Samples));
                best = Math.Min(best, total / Sources);
            }

            expected += best / Batch;
        }

        Assert.Equal(expected, loss.ComputeTapeLoss(estimate, reference)[0], 8);
    }

    [Fact]
    public void ManySources_AreMatchedWithoutEnumeratingPermutations()
    {
        // 10 sources is 3.6 million permutations; matched by assignment it is 100 pair scores.
        const int Sources = 10, Samples = 24;
        var loss = new NegativeSiSnrLoss<double>();
        var reference = Random(13, 1, Sources, Samples);
        var noise = Random(14, 1, Sources, Samples);
        var shuffled = new Tensor<double>(reference.Shape.ToArray());
        var inOrder = new Tensor<double>(reference.Shape.ToArray());
        for (int s = 0; s < Sources; s++)
        {
            int target = (s * 3 + 1) % Sources;   // a fixed derangement-style shuffle of the estimate order
            for (int t = 0; t < Samples; t++)
            {
                double value = reference[0, s, t] + 0.05 * noise[0, s, t];
                inOrder[0, s, t] = value;
                shuffled[0, target, t] = value;
            }
        }

        var clock = System.Diagnostics.Stopwatch.StartNew();
        double ordered = loss.ComputeTapeLoss(inOrder, reference)[0];
        double permuted = loss.ComputeTapeLoss(shuffled, reference)[0];
        clock.Stop();

        Assert.Equal(ordered, permuted, 8);
        Assert.True(ordered < -20.0, $"Close estimates should score well above 20 dB; got {-ordered:F2} dB.");
        Assert.True(clock.Elapsed.TotalSeconds < 10, $"Matching 10 sources took {clock.Elapsed.TotalSeconds:F1} s.");
    }

    private static Vector<double> Row(Tensor<double> tensor, int b, int s, int samples)
    {
        var values = new double[samples];
        for (int t = 0; t < samples; t++) values[t] = tensor[b, s, t];
        return new Vector<double>(values);
    }

    private static System.Collections.Generic.IEnumerable<int[]> Orders(int count)
    {
        if (count == 1)
        {
            yield return new[] { 0 };
            yield break;
        }

        foreach (var rest in Orders(count - 1))
        {
            for (int position = 0; position < count; position++)
            {
                var order = new int[count];
                for (int k = 0, r = 0; k < count; k++) order[k] = k == position ? count - 1 : rest[r++];
                yield return order;
            }
        }
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