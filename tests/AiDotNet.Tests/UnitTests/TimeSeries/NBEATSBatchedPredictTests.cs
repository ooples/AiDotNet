using System;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// NBEATSModel.Predict runs rows of exactly LookbackWindow values through the stack as batched chunked forwards. It
/// must return exactly what PredictSingle returns row by row, including a row count that spans more than one chunk.
/// </summary>
public sealed class NBEATSBatchedPredictTests
{
    [Theory]
    [InlineData(17)]
    [InlineData(1100)]
    [Trait("category", "unit")]
    public void BatchedPredict_MatchesPerRowPredictSingle(int rows)
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 12, LagOrder = 12, ForecastHorizon = 2, NumStacks = 2, NumBlocksPerStack = 2,
            HiddenLayerSize = 16, Epochs = 3, BatchSize = 16, Seed = 3,
        };
        var model = new NBEATSModel<double>(options);
        int n = 80;
        var trainX = new Matrix<double>(n, options.LookbackWindow);
        var trainY = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++) trainX[i, j] = Math.Sin((i + j) * 0.3);
            trainY[i] = Math.Sin((i + options.LookbackWindow) * 0.3) + 0.05 * Math.Cos(i);
        }
        model.Train(trainX, trainY);

        var rng = new Random(rows);
        var input = new Matrix<double>(rows, options.LookbackWindow);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < options.LookbackWindow; j++) input[i, j] = rng.NextDouble() * 2 - 1;

        var batched = model.Predict(input);
        Assert.Equal(rows, batched.Length);
        for (int i = 0; i < rows; i++)
        {
            double single = model.PredictSingle(input.GetRow(i));
            Assert.True(Math.Abs(single - batched[i]) <= 1e-10 * Math.Max(1, Math.Abs(single)),
                $"row {i}: batched {batched[i]:R} vs per-row {single:R}");
        }
    }
}
