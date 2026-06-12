using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Paper-fidelity / non-divergence guard for <see cref="CTGANGenerator{T}"/>
/// (Xu et al. 2019). The Synthetic POC review found CTGAN "diverges on small data,
/// worse with more epochs". Root cause: the generator and discriminator shared one
/// AdamOptimizer instance, whose single flat (_m,_v) moment buffer was reallocated
/// (and the timestep reset to 1) on every alternating Step because the two networks
/// have different parameter counts — so Adam never built valid second moments and
/// the step size stayed miscalibrated, compounding every epoch. The fix uses
/// separate WGAN-GP-configured optimizers (β1=0.5, β2=0.9). This test trains for
/// enough epochs that the old shared-optimizer code blew up, and asserts the
/// trained model produces stable, in-distribution, bimodal output.
/// </summary>
public class CtganPaperFidelityTests
{
    private const int Seed = 1729;

    [Fact]
    public void Ctgan_TrainsStably_AndRecoversBimodalDistribution()
    {
        // Real data: a strongly bimodal continuous column (N(-5,0.5) ∪ N(5,0.5))
        // plus a binary categorical column correlated with the mode. This is the
        // multimodal-tabular case CTGAN's VGM + conditional machinery exists for.
        const int perMode = 150;
        int rows = perMode * 2;
        var rng = new Random(Seed);
        var data = new Matrix<double>(rows, 2);
        for (int i = 0; i < perMode; i++)
        {
            data[i, 0] = -5.0 + 0.5 * SampleNormal(rng);
            data[i, 1] = 0; // category 0 dominates the low mode
        }
        for (int i = 0; i < perMode; i++)
        {
            data[perMode + i, 0] = 5.0 + 0.5 * SampleNormal(rng);
            data[perMode + i, 1] = 1; // category 1 dominates the high mode
        }

        var columns = new List<ColumnMetadata>
        {
            new("X", ColumnDataType.Continuous, columnIndex: 0),
            new("C", ColumnDataType.Categorical, new[] { "lo", "hi" }, columnIndex: 1),
        };

        var options = new CTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = new[] { 64, 64 },
            DiscriminatorDimensions = new[] { 64, 64 },
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 10,
            DiscriminatorSteps = 1,
            LearningRate = 1e-3,
        };

        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            inputSize: 2,
            outputSize: 2);

        var ctgan = new CTGANGenerator<double>(arch, options);

        // 60 epochs — well past the point the shared-optimizer code diverged.
        ctgan.Fit(data, columns, epochs: 60);
        var synth = ctgan.Generate(400);

        // (1) NON-DIVERGENCE: every generated value finite and in a sane range.
        //     The old code produced NaN/Inf or values exploding to ±1e6 here.
        double realMin = -8, realMax = 8;
        for (int i = 0; i < synth.Rows; i++)
        {
            double x = synth[i, 0];
            Assert.False(double.IsNaN(x) || double.IsInfinity(x), $"row {i}: non-finite generated value {x} (CTGAN diverged).");
            Assert.InRange(x, realMin * 3, realMax * 3); // generous, but catches explosion
        }

        // (2) NOT COLLAPSED: generated std must be in a reasonable band of the real
        //     std (~5.0 for this bimodal column). Mode collapse would crush it.
        double mean = 0; for (int i = 0; i < synth.Rows; i++) mean += synth[i, 0]; mean /= synth.Rows;
        double var0 = 0; for (int i = 0; i < synth.Rows; i++) { double d = synth[i, 0] - mean; var0 += d * d; }
        double std = Math.Sqrt(var0 / synth.Rows);
        Assert.True(std > 1.5, $"generated std {std:F2} too small — mode collapse (real std ≈ 5).");

        // (3) BIMODAL: mass on BOTH modes. A collapsed generator sits on one.
        int low = 0, high = 0;
        for (int i = 0; i < synth.Rows; i++)
        {
            if (synth[i, 0] < -1.5) low++;
            else if (synth[i, 0] > 1.5) high++;
        }
        Assert.True(low > synth.Rows / 10, $"too few low-mode samples ({low}) — generator ignored a mode.");
        Assert.True(high > synth.Rows / 10, $"too few high-mode samples ({high}) — generator ignored a mode.");

        // (4) Categorical column stays a valid category (0 or 1).
        for (int i = 0; i < synth.Rows; i++)
        {
            int c = (int)Math.Round(synth[i, 1]);
            Assert.InRange(c, 0, 1);
        }
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
