using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>Batched Predict (grouped by window length) must equal per-row prediction, in-sample and out-of-sample.</summary>
public class AutoformerBatchedPredictTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BatchedPredict_MatchesPerRowForecasts(bool inSample)
    {
        var model = new AutoformerModel<double>(new AutoformerOptions<double>
        {
            LookbackWindow = 10, ForecastHorizon = 1, EmbeddingDim = 8, NumEncoderLayers = 1, NumDecoderLayers = 1,
            Epochs = 2, UseEarlyStopping = false, Seed = 6,
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
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
            input = new Matrix<double>(17, 12);       // out-of-sample rows longer than the lookback
            for (int i = 0; i < 17; i++) for (int j = 0; j < 12; j++) input[i, j] = Math.Sin(0.3 * (i + j)) + 0.1 * rng.NextDouble();
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
            else if (inSample)
            {
                // The first observation has no history: its prediction is the training mean.
                double mean = 0;
                for (int t = 0; t < n; t++) mean += y[t];
                expected = mean / n;
            }
            else expected = model.PredictSingle(input.GetRow(i));
            Assert.True(Math.Abs(expected - batched[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected)),
                $"row {i}: batched {batched[i]:R} vs per-row {expected:R}");
        }
    }
}
