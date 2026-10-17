using System;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// InformerModel.Predict runs one batched forward per (window length, chunk) instead of one per row. It must equal
/// per-row prediction in-sample (growing training-series windows) and out-of-sample (rows shorter and longer than the
/// lookback), including a row count spanning more than one chunk.
/// </summary>
public class InformerBatchedPredictTests
{
    [Theory]
    [InlineData(true, 0, 0)]
    [InlineData(false, 17, 7)]
    [InlineData(false, 17, 12)]
    [InlineData(false, 300, 10)]
    public void BatchedPredict_MatchesPerRowForecasts(bool inSample, int rows, int width)
    {
        var model = new InformerModel<double>(new InformerOptions<double>
        {
            LookbackWindow = 10, ForecastHorizon = 1, EmbeddingDim = 8, NumAttentionHeads = 2, NumEncoderLayers = 2,
            NumDecoderLayers = 1, Epochs = 2, UseEarlyStopping = false, Seed = 6,
        });
        int n = 60;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++) { x[i, 0] = i; y[i] = Math.Sin(i * 0.3) + 0.02 * i; }
        model.Train(x, y);

        Matrix<double> input;
        if (inSample)
        {
            // In-sample (row count == series length) with TWO columns: a single column is treated as a time
            // index and answered by the calibration path before any window is built.
            input = new Matrix<double>(n, 2);
            for (int i = 0; i < n; i++) { input[i, 0] = i; input[i, 1] = y[i]; }
        }
        else
        {
            var rng = new Random(rows + width);
            input = new Matrix<double>(rows, width);
            for (int i = 0; i < rows; i++) for (int j = 0; j < width; j++) input[i, j] = Math.Sin(0.3 * (i + j)) + 0.1 * rng.NextDouble();
        }
        var batched = model.Predict(input);
        for (int i = 0; i < input.Rows; i++)
        {
            double expected;
            if (inSample && i > 0)
            {
                int w = Math.Min(10, i);
                var window = new Vector<double>(w);
                for (int t = 0; t < w; t++) window[t] = y[i - w + t];
                expected = model.PredictSingle(window);
            }
            else expected = model.PredictSingle(input.GetRow(i));
            Assert.True(Math.Abs(expected - batched[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected)),
                $"row {i}: batched {batched[i]:R} vs per-row {expected:R}");
        }
    }
}
