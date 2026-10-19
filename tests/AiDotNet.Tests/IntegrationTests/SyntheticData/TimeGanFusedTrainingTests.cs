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
/// TimeGAN's three phases, its joint generator/supervisor phase included, train on the shared fused step (#1804).
/// With compilation on, a fit must leave the model where the eager fit leaves it: the joint phase's noise and real
/// latent pairs travel in each step's data, so a plan that replayed the first batch's would land elsewhere.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class TimeGanFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; a step on another batch's data moves weights by far more.
    private const double RelativeTolerance = 1e-5;

    private static Vector<double> Fit(bool compile)
    {
        var data = new Matrix<double>(24, 2);
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
        var generator = new TimeGANGenerator<double>(
            new NeuralNetworkArchitecture<double>(2, 2, NetworkComplexity.Simple),
            new TimeGANOptions<double> { Seed = 5, HiddenDimension = 6, NumLayers = 1, SequenceLength = 3, BatchSize = 4 });

        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            // Six epochs: two per phase. The total is split across the three phases, so fewer than three never
            // reaches the joint phase.
            generator.Fit(data, columns, epochs: 6);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
        return generator.GetParameters();
    }

    [Fact(Timeout = 300000)]
    public async Task FusedFit_LandsWhereTheEagerFitDoes()
    {
        await Task.Yield();
        var eager = Fit(compile: false);
        var fused = Fit(compile: true);

        Assert.Equal(eager.Length, fused.Length);
        double scale = 0.0, worst = 0.0;
        for (int i = 0; i < eager.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(eager[i]));
            worst = Math.Max(worst, Math.Abs(eager[i] - fused[i]));
        }
        Assert.True(worst <= RelativeTolerance * Math.Max(scale, 1.0),
            $"fused TimeGAN fit diverged from the eager fit by {worst:G6} (parameter scale {scale:G6}).");
    }
}
