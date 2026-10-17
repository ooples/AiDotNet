using System;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// TemporalFusionTransformer's parameter vector must carry every trained weight: copying a trained model's
/// GetParameters into a fresh model must reproduce its forecasts. The four GRNs and the Q/K/V/O projections used to be
/// missing from ParameterCount/GetParameters/SetParameters (931 values for a ~22k-weight model), so the copy kept its
/// random initialization for most of the network.
/// </summary>
public class TftParameterRoundTripTests
{
    [Fact]
    public void GetSetParameters_CarriesEveryTrainedWeight()
    {
        TemporalFusionTransformerOptions<double> Options() => new()
        {
            LookbackWindow = 10, HiddenSize = 8, NumAttentionHeads = 2, ForecastHorizon = 2, Epochs = 3,
            UseEarlyStopping = false, Seed = 6,
        };
        int n = 60;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++) { x[i, 0] = i; y[i] = Math.Sin(i * 0.3) + 0.02 * i; }
        var trained = new TemporalFusionTransformer<double>(Options());
        trained.Train(x, y);

        // 8-wide: embedding 16, Q/K/V/O 4x64, quantile head 8x6+6, four GRNs of 4x(64+8)+2x8 = 304 each.
        long trainable = 16 + 256 + 54 + 4 * 304;
        Assert.True(trained.ParameterCount >= trainable,
            $"ParameterCount {trained.ParameterCount} < {trainable} trainable weights");
        Assert.Equal(trained.ParameterCount, trained.GetParameters().Length);

        var fresh = new TemporalFusionTransformer<double>(new TemporalFusionTransformerOptions<double>(Options()) { Seed = 99 });
        fresh.Train(x, y);   // fits normalization statistics and the training series; weights are then overwritten
        fresh.SetParameters(trained.GetParameters());

        var window = new Vector<double>(10);
        for (int t = 0; t < 10; t++) window[t] = y[40 + t];
        double expected = trained.PredictSingle(window), actual = fresh.PredictSingle(window);
        Assert.True(Math.Abs(expected - actual) <= 1e-12 * Math.Max(1, Math.Abs(expected)),
            $"copied model predicts {actual:R}, trained model {expected:R}");
    }
}
