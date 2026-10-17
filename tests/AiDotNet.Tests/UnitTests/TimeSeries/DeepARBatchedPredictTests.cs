using System;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// DeepARModel.Predict over many rows unrolls the LSTM once for all rows (columns of the [H, B] state) instead of
/// once per row. It must return exactly what PredictSingle returns row by row: for windows equal to, shorter than
/// (left-padded) and longer than the lookback, under every distribution head.
/// </summary>
public sealed class DeepARBatchedPredictTests
{
    [Theory]
    [InlineData("Gaussian", 12)]
    [InlineData("Gaussian", 7)]
    [InlineData("Gaussian", 15)]
    [InlineData("StudentT", 12)]
    [InlineData("Spline", 9)]
    [Trait("category", "unit")]
    public void BatchedPredict_MatchesPerRowPredictSingle(string likelihood, int windowLength)
    {
        var options = new DeepAROptions<double>
        {
            LookbackWindow = 12, ForecastHorizon = 1, HiddenSize = 16, NumLayers = 2, Epochs = 3, BatchSize = 8,
            LikelihoodType = likelihood, StudentTDegreesOfFreedom = 4.0, Seed = 3,
        };
        var model = new DeepARModel<double>(options);
        int n = 80;
        var trainX = new Matrix<double>(n, options.LookbackWindow);
        var trainY = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++) trainX[i, j] = Math.Sin((i + j) * 0.3);
            trainY[i] = Math.Sin((i + options.LookbackWindow) * 0.3) + 0.05 * Math.Cos(i);
        }
        model.Train(trainX, trainY);

        var rng = new Random(windowLength);
        var input = new Matrix<double>(17, windowLength);
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
