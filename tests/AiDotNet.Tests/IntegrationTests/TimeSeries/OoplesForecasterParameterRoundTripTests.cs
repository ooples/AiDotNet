using System;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Every deep forecaster Ooples trains must carry ALL of its trained weights in its parameter vector: copying a
/// trained model's GetParameters into a differently seeded model must reproduce its forecasts. A weight held in a
/// field the parameter registry does not see trains normally but is silently dropped by clone, checkpoint and save
/// (TemporalFusionTransformer lost its GRNs and attention projections this way).
/// </summary>
public class OoplesForecasterParameterRoundTripTests
{
    private static IFullModel<double, Matrix<double>, Vector<double>> Make(string name, int seed) => name switch
    {
        "deepar" => new DeepARModel<double>(new DeepAROptions<double>
            { LookbackWindow = 10, ForecastHorizon = 1, HiddenSize = 8, NumLayers = 2, Epochs = 2, Seed = seed }),
        "nbeats" => new NBEATSModel<double>(new NBEATSModelOptions<double>
        {
            LookbackWindow = 10, LagOrder = 10, ForecastHorizon = 1, NumStacks = 2, NumBlocksPerStack = 1,
            HiddenLayerSize = 8, Epochs = 2, Seed = seed,
        }),
        "informer" => new InformerModel<double>(new InformerOptions<double>
        {
            LookbackWindow = 10, ForecastHorizon = 1, EmbeddingDim = 8, NumAttentionHeads = 2, NumEncoderLayers = 2,
            NumDecoderLayers = 1, Epochs = 2, Seed = seed,
        }),
        "chronos" => new ChronosFoundationModel<double>(new ChronosOptions<double>
            { ContextLength = 10, ForecastHorizon = 1, EmbeddingDim = 8, NumLayers = 1, NumHeads = 2, Epochs = 1, Seed = seed }),
        "tft" => new TemporalFusionTransformer<double>(new TemporalFusionTransformerOptions<double>
            { LookbackWindow = 10, ForecastHorizon = 1, HiddenSize = 8, NumAttentionHeads = 2, Epochs = 2, Seed = seed }),
        "nhits" => new NHiTSModel<double>(new NHiTSOptions<double>
        {
            LookbackWindow = 10, ForecastHorizon = 1, HiddenLayerSize = 8, Epochs = 2, Seed = seed,
            PoolingKernelSizes = new[] { 5, 2, 1 },
        }),
        "autoformer" => new AutoformerModel<double>(new AutoformerOptions<double>
        {
            LookbackWindow = 10, ForecastHorizon = 1, EmbeddingDim = 8, NumEncoderLayers = 1, NumDecoderLayers = 1,
            Epochs = 2, Seed = seed,
        }),
        "dlinear" => new DLinearModel<double>(new DLinearOptions<double>
            { LookbackWindow = 10, ForecastHorizon = 1, Epochs = 2, Seed = seed }),
        _ => throw new ArgumentException(name),
    };

    [Theory]
    [InlineData("deepar")]
    [InlineData("nbeats")]
    [InlineData("informer")]
    [InlineData("chronos")]
    [InlineData("tft")]
    [InlineData("nhits")]
    [InlineData("autoformer")]
    [InlineData("dlinear")]
    public void GetSetParameters_ReproducesTrainedForecasts(string name)
    {
        int n = 60;
        var x = new Matrix<double>(n, 10);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            y[i] = Math.Sin(i * 0.3) + 0.02 * i;
            for (int j = 0; j < 10; j++) x[i, j] = Math.Sin((i - 10 + j) * 0.3) + 0.02 * (i - 10 + j);
        }
        var trained = Make(name, 6);
        trained.Train(x, y);
        var copy = Make(name, 99);
        copy.Train(x, y);                        // same data statistics; weights are then overwritten
        copy.SetParameters(trained.GetParameters());

        var probe = new Matrix<double>(5, 10);
        for (int i = 0; i < 5; i++) for (int j = 0; j < 10; j++) probe[i, j] = x[40 + i, j];
        var expected = trained.Predict(probe);
        var actual = copy.Predict(probe);
        for (int i = 0; i < 5; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])),
                $"{name} row {i}: copied model {actual[i]:R} vs trained {expected[i]:R} " +
                $"(ParameterCount {trained.ParameterCount})");
    }
}
