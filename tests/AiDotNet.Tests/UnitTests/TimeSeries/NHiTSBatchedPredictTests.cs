using System;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// NHiTSModel.Predict runs full-lookback rows (and shorter rows, which PredictSingle replaces with the training tail)
/// as batched forwards in chunks. It must return exactly what PredictSingle returns row by row, including rows longer
/// than the lookback (per-row path) and a row count that spans more than one chunk.
/// </summary>
public sealed class NHiTSBatchedPredictTests
{
    [Theory]
    [InlineData(12, 17)]
    [InlineData(7, 17)]
    [InlineData(15, 17)]
    [InlineData(12, 1100)]
    [Trait("category", "unit")]
    public void BatchedPredict_MatchesPerRowPredictSingle(int windowLength, int rows)
    {
        var options = new NHiTSOptions<double>
        {
            LookbackWindow = 12, ForecastHorizon = 1, HiddenLayerSize = 32, Epochs = 3, BatchSize = 8, Seed = 3,
            PoolingKernelSizes = new[] { 5, 2, 1 },
        };
        var model = new NHiTSModel<double>(options);
        int n = 80;
        var trainX = new Matrix<double>(n, options.LookbackWindow);
        var trainY = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++) trainX[i, j] = Math.Sin((i + j) * 0.3);
            trainY[i] = Math.Sin((i + options.LookbackWindow) * 0.3) + 0.05 * Math.Cos(i);
        }
        model.Train(trainX, trainY);

        var rng = new Random(windowLength + rows);
        var input = new Matrix<double>(rows, windowLength);
        for (int i = 0; i < input.Rows; i++)
            for (int j = 0; j < windowLength; j++) input[i, j] = rng.NextDouble() * 2 - 1;

        var batched = model.Predict(input);
        Assert.Equal(input.Rows, batched.Length);
        for (int i = 0; i < input.Rows; i++)
        {
            double single = model.PredictSingle(input.GetRow(i));
            Assert.True(Math.Abs(single - batched[i]) <= 1e-12 * Math.Max(1, Math.Abs(single)),
                $"row {i}: batched {batched[i]:R} vs per-row {single:R}");
        }
    }
}
