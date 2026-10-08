using System;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// ChronosFoundationModel trains with hand-written backprop (no GradientTape) inside the TensorArena that
/// TimeSeriesModelBase.Train opens around TrainCore. Without a per-sample arena reset every Engine op took a fresh ring
/// buffer that lived until training ended (74 GB on an Ooples-sized fit). The arena's peak working set must now stay
/// bounded as the training set grows, and recycling must not change what the model learns.
/// </summary>
public sealed class ChronosArenaRecycleTests
{
    private static ChronosFoundationModel<double> Train(int rows, int epochs = 1)
    {
        var model = new ChronosFoundationModel<double>(new ChronosOptions<double>
        {
            Seed = 11, ContextLength = 12, ForecastHorizon = 1, EmbeddingDim = 16, NumLayers = 2, NumHeads = 2,
            Epochs = epochs, UseEarlyStopping = false,
        });
        var x = new Matrix<double>(rows, 1);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++) { x[i, 0] = i; y[i] = Math.Sin(i * 0.2) + 0.01 * i; }
        model.Train(x, y);
        return model;
    }

    [Fact]
    [Trait("category", "unit")]
    public void ArenaPeak_DoesNotGrowWithTrainingSetSize()
    {
        Train(60);
        long small = TensorArena.LastDisposedPeakBackingBytes;
        Train(240);
        long large = TensorArena.LastDisposedPeakBackingBytes;
        // 4x the samples: with per-sample recycling the arena's peak is one sample's working set either way.
        Assert.True(large <= small * 1.5 + 1_000_000,
            $"arena peak grew with the training set: {small:N0} bytes at 60 rows vs {large:N0} at 240 rows");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Recycling_DoesNotChangeTheTrainedModel()
    {
        // The same seed trained twice in a row must forecast identically; a recycled tensor that something still
        // referenced (an aliased accumulator, a stale cache) would corrupt the second run's updates differently.
        var a = Train(120, epochs: 2);
        var b = Train(120, epochs: 2);
        var history = new Vector<double>(24);
        for (int i = 0; i < history.Length; i++) history[i] = Math.Sin(i * 0.2);
        var fa = a.Forecast(history, 3);
        var fb = b.Forecast(history, 3);
        for (int i = 0; i < 3; i++) Assert.Equal(fa[i], fb[i]);
        for (int i = 0; i < 3; i++) Assert.False(double.IsNaN(fa[i]) || double.IsInfinity(fa[i]));
    }
}
