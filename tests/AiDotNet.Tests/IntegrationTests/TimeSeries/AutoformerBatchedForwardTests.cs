using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// AutoformerModel.ForwardBatched (one tape per mini-batch) against the per-sample ForwardCore it replaces in
/// training: the same forecasts row by row, and the batch-mean-loss gradient equal to the mean of the per-sample
/// gradients for every trainable tensor -- including each sample's own data-dependent top-k delay selection.
/// </summary>
public class AutoformerBatchedForwardTests
{
    private static AutoformerModel<double> Model(int horizon) => new(new AutoformerOptions<double>
    {
        LookbackWindow = 12, ForecastHorizon = horizon, EmbeddingDim = 8, NumEncoderLayers = 2, NumDecoderLayers = 1,
        Epochs = 1, UseEarlyStopping = false, Seed = 4,
    });

    private static double[][] Windows(int count, int length, int seed)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(s =>
            Enumerable.Range(0, length).Select(t => Math.Sin(0.5 * (s + t)) + 0.4 * (rng.NextDouble() - 0.5)).ToArray()).ToArray();
    }

    [Theory]
    [InlineData(1, 1, 12)]
    [InlineData(6, 1, 12)]
    [InlineData(5, 3, 12)]
    [InlineData(4, 1, 1)]
    [InlineData(4, 1, 2)]
    [InlineData(4, 2, 3)]
    public void BatchedForward_EqualsPerSampleForward(int batch, int horizon, int length)
    {
        var model = Model(horizon);
        var windows = Windows(batch, length, batch * 10 + horizon);
        var stacked = new Tensor<double>(new[] { batch, length }, new Vector<double>(windows.SelectMany(w => w).ToArray()));
        var batched = model.ForwardBatched(stacked);
        for (int s = 0; s < batch; s++)
        {
            var single = model.ForwardCore(new Vector<double>(windows[s]));
            for (int h = 0; h < horizon; h++)
                Assert.True(Math.Abs(single[h] - batched[s, h]) <= 1e-10 * Math.Max(1, Math.Abs(single[h])),
                    $"sample {s} step {h}: batched {batched[s, h]:R} vs per-sample {single[h]:R}");
        }
    }

    [Fact]
    public void BatchMeanLossGradient_EqualsMeanOfPerSampleGradients()
    {
        const int batch = 5, horizon = 2;
        var model = Model(horizon);
        var windows = Windows(batch, 12, 77);
        var targets = Windows(batch, horizon, 78);
        var parameters = model.CollectTrainableParameters();

        // Reference: per-sample MSE gradients (one tape each), averaged.
        var mean = parameters.Select(p => new double[p.Length]).ToArray();
        for (int s = 0; s < batch; s++)
        {
            using var tape = new GradientTape<double>();
            var pred = model.ForwardCore(new Vector<double>(windows[s]));                   // [H, 1]
            var target = new Tensor<double>(new[] { horizon, 1 }, new Vector<double>(targets[s]));
            var diff = Engine().TensorSubtract(pred, target);
            var loss = Engine().ReduceMean(Engine().TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
            var g = tape.ComputeGradients(loss, parameters.ToArray());
            for (int p = 0; p < parameters.Count; p++)
                if (g.TryGetValue(parameters[p], out var gp)) { var a = gp.ToArray(); for (int i = 0; i < a.Length; i++) mean[p][i] += a[i] / batch; }
        }

        double[][] batched;
        using (var tape = new GradientTape<double>())
        {
            var stacked = new Tensor<double>(new[] { batch, 12 }, new Vector<double>(windows.SelectMany(w => w).ToArray()));
            var pred = model.ForwardBatched(stacked);                                        // [B, H]
            var target = new Tensor<double>(new[] { batch, horizon }, new Vector<double>(targets.SelectMany(t => t).ToArray()));
            var diff = Engine().TensorSubtract(pred, target);
            var loss = Engine().ReduceMean(Engine().TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
            var g = tape.ComputeGradients(loss, parameters.ToArray());
            batched = parameters.Select(p => g.TryGetValue(p, out var gp) ? gp.ToArray() : new double[p.Length]).ToArray();
        }

        for (int p = 0; p < parameters.Count; p++)
        {
            double maxAbs = mean[p].Select(Math.Abs).DefaultIfEmpty(0).Max();
            double maxErr = mean[p].Zip(batched[p], (a, b) => Math.Abs(a - b)).DefaultIfEmpty(0).Max();
            Assert.True(maxErr <= 1e-9 * Math.Max(1, maxAbs), $"parameter {p}: max |error| {maxErr:E3} vs max |ref| {maxAbs:E3}");
        }
    }

    private static AiDotNet.Tensors.Engines.IEngine Engine() => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
}

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
            var rng = new Random(3);
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
            else expected = model.PredictSingle(input.GetRow(i));
            Assert.True(Math.Abs(expected - batched[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected)),
                $"row {i}: batched {batched[i]:R} vs per-row {expected:R}");
        }
    }
}
