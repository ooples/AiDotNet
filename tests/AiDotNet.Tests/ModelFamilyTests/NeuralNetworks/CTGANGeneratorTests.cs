using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for <see cref="CTGANGenerator{T}"/> (Xu et al. 2019,
/// "Modeling Tabular Data using Conditional GAN"). Follows the
/// <see cref="TableGANGeneratorTests"/> pattern: CTGAN's real training runs
/// through <see cref="CTGANGenerator{T}.Fit"/> against a tabular matrix + column
/// metadata, not the <c>Train(Tensor, Tensor)</c> surface the auto-generated
/// <c>TestScaffoldGenerator</c> invariants exercise. Being named
/// <c>CTGANGeneratorTests</c> suppresses that auto-generated stub and lets this
/// scaffold cover the model's real surface — construction, Predict, the
/// Fit + Generate path — plus the paper-faithfulness invariants that pin the
/// Synthetic POC's findings (divergence on small data, worse with more epochs).
/// </summary>
public class CTGANGeneratorTests
{
    private const int Seed = 1729;

    private static CTGANGenerator<double> CreateGenerator(CTGANOptions<double>? options = null)
    {
        options ??= new CTGANOptions<double>
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
        return new CTGANGenerator<double>(arch, options);
    }

    /// <summary>
    /// Bimodal dataset: a strongly separated continuous column (N(-5,0.5) ∪
    /// N(5,0.5)) + a binary categorical column correlated with the mode — the
    /// multimodal-tabular case CTGAN's VGM + conditional machinery exists for.
    /// </summary>
    private static (Matrix<double> data, ColumnMetadata[] columns) BuildBimodalDataset(int perMode = 150)
    {
        int rows = perMode * 2;
        var rng = new Random(Seed);
        var data = new Matrix<double>(rows, 2);
        for (int i = 0; i < perMode; i++) { data[i, 0] = -5.0 + 0.5 * SampleNormal(rng); data[i, 1] = 0; }
        for (int i = 0; i < perMode; i++) { data[perMode + i, 0] = 5.0 + 0.5 * SampleNormal(rng); data[perMode + i, 1] = 1; }
        var columns = new[]
        {
            new ColumnMetadata("X", ColumnDataType.Continuous, columnIndex: 0),
            new ColumnMetadata("C", ColumnDataType.Categorical, new[] { "lo", "hi" }, columnIndex: 1),
        };
        return (data, columns);
    }

    [Fact]
    public void Constructor_DefaultOptions_DoesNotThrow()
    {
        using var gen = CreateGenerator();
        Assert.NotNull(gen);
        Assert.False(gen.IsFitted);
    }

    [Fact]
    public void Generate_AfterFit_ProducesFiniteSyntheticRows()
    {
        var (data, columns) = BuildBimodalDataset(perMode: 40);
        using var gen = CreateGenerator();
        gen.Fit(data, columns, epochs: 5);

        var synthetic = gen.Generate(numSamples: 16);

        Assert.Equal(16, synthetic.Rows);
        Assert.Equal(columns.Length, synthetic.Columns);
        for (int r = 0; r < synthetic.Rows; r++)
            for (int c = 0; c < synthetic.Columns; c++)
                Assert.False(double.IsNaN(synthetic[r, c]) || double.IsInfinity(synthetic[r, c]),
                    $"Synthetic[{r},{c}] not finite: {synthetic[r, c]}");
    }

    /// <summary>
    /// PAPER INVARIANT (the POC's headline finding): CTGAN must train stably and
    /// recover the bimodal distribution. The pre-fix code shared one AdamOptimizer
    /// across generator and discriminator, whose flat moment buffer reallocated and
    /// reset the timestep on every alternating step (different param counts) — so it
    /// diverged worse with more epochs. This trains 60 epochs (well past the old
    /// blow-up point) and asserts non-divergence, no mode collapse, and bimodality.
    /// </summary>
    [Fact]
    public void Ctgan_TrainsStably_AndRecoversBimodalDistribution()
    {
        var (data, columns) = BuildBimodalDataset(perMode: 150);
        using var gen = CreateGenerator();

        gen.Fit(data, columns, epochs: 60);
        var synth = gen.Generate(400);

        double mean = 0;
        for (int i = 0; i < synth.Rows; i++)
        {
            double x = synth[i, 0];
            Assert.False(double.IsNaN(x) || double.IsInfinity(x), $"row {i}: non-finite value {x} (CTGAN diverged).");
            Assert.InRange(x, -24, 24); // generous; catches explosion
            mean += x;
        }
        mean /= synth.Rows;

        double var0 = 0;
        for (int i = 0; i < synth.Rows; i++) { double d = synth[i, 0] - mean; var0 += d * d; }
        double std = Math.Sqrt(var0 / synth.Rows);
        Assert.True(std > 1.5, $"generated std {std:F2} too small — mode collapse (real std ≈ 5).");

        int low = 0, high = 0;
        for (int i = 0; i < synth.Rows; i++)
        {
            if (synth[i, 0] < -1.5) low++;
            else if (synth[i, 0] > 1.5) high++;
        }
        Assert.True(low > synth.Rows / 10, $"too few low-mode samples ({low}) — a mode was ignored.");
        Assert.True(high > synth.Rows / 10, $"too few high-mode samples ({high}) — a mode was ignored.");

        for (int i = 0; i < synth.Rows; i++)
            Assert.InRange((int)Math.Round(synth[i, 1]), 0, 1);
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
