using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// MisGAN's four adversarial steps (both critics, both generators) train on the shared fused step (#1804), each with
/// its rows, masks and noise in the step's data, and the WGAN weight clip happens in place. With compilation on, a
/// fit must leave the model where the eager fit leaves it, and two seeded fits must agree.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class MisGanFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; a step on another sample's data moves weights by far more.
    private const double RelativeTolerance = 1e-5;

    private static Vector<double> Fit(bool compile)
    {
        var data = new Matrix<double>(12, 2);
        for (int i = 0; i < data.Rows; i++)
        {
            data[i, 0] = Math.Sin(0.4 * i);
            data[i, 1] = Math.Cos(0.3 * i) * 0.5;
        }
        var columns = new List<ColumnMetadata>
        {
            new("A", ColumnDataType.Continuous, columnIndex: 0),
            new("B", ColumnDataType.Continuous, columnIndex: 1),
        };
        var generator = new MisGANGenerator<double>(
            new NeuralNetworkArchitecture<double>(2, 2, NetworkComplexity.Simple),
            new MisGANOptions<double>
            {
                Seed = 9, EmbeddingDimension = 4, HiddenDimensions = [8], BatchSize = 4, VGMModes = 2,
                DiscriminatorDropout = 0.0, DiscriminatorSteps = 1,
            });

        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            generator.Fit(data, columns, epochs: 2);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
        return generator.GetParameters();
    }

    private static double Worst(Vector<double> a, Vector<double> b, out double scale)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        scale = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(a[i]));
            worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        }
        return worst;
    }

    [Fact(Timeout = 300000)]
    public async Task TwoSeededFits_Agree()
    {
        await Task.Yield();
        double worst = Worst(Fit(compile: false), Fit(compile: false), out double scale);
        Assert.True(worst == 0.0, $"two seeded eager fits differ by {worst:G6} (parameter scale {scale:G6}).");
    }

    [Fact(Timeout = 300000)]
    public async Task FusedFit_LandsWhereTheEagerFitDoes()
    {
        await Task.Yield();
        var eager = Fit(compile: false);
        var fused = Fit(compile: true);
        double worst = Worst(eager, fused, out double scale);
        Assert.True(worst <= RelativeTolerance * Math.Max(scale, 1.0),
            $"fused MisGAN fit diverged from the eager fit by {worst:G6} (parameter scale {scale:G6}).");
    }
}
