using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Chronos trains on a batched tape gradient (ComputeBatchGradientsTape). It must equal central finite differences of
/// the model's own per-token training loss (ReferenceSampleLoss) on every parameter group, and a batch of windows must
/// yield the sum of the single-window gradients. (The hand-written per-sample backprop this replaced disagreed with
/// finite differences on 66 of 72 transformer-parameter probes.)
/// </summary>
public sealed class ChronosBatchedGradientParityTests
{
    private readonly ITestOutputHelper _out;
    public ChronosBatchedGradientParityTests(ITestOutputHelper output) => _out = output;

    private static ChronosFoundationModel<double> Model() => new(new ChronosOptions<double>
    {
        Seed = 5, ContextLength = 8, ForecastHorizon = 1, EmbeddingDim = 8, NumLayers = 2, NumHeads = 2,
        VocabularySize = 64, Epochs = 1, UseEarlyStopping = false,
    });

    private static (List<Vector<double>> Windows, List<double> Targets) Samples(int windowLength, int count, int seed)
    {
        var rng = new Random(seed);
        var windows = new List<Vector<double>>(); var targets = new List<double>();
        for (int s = 0; s < count; s++)
        {
            var w = new Vector<double>(windowLength);
            for (int t = 0; t < windowLength; t++) w[t] = Math.Sin(0.4 * (s + t)) + 0.3 * rng.NextDouble();
            windows.Add(w);
            targets.Add(Math.Sin(0.4 * (s + windowLength)) + 0.3 * rng.NextDouble());
        }
        return (windows, targets);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(3)]
    [Trait("category", "unit")]
    public void TapeGradient_MatchesFiniteDifferencesOfTheModelLoss(int windowLength)
    {
        var model = Model();
        var (windows, targets) = Samples(windowLength, 1, windowLength);
        var tape = model.ComputeBatchGradientsTape(windows, targets);
        const double eps = 1e-6;
        int probes = 0;
        foreach (var (key, param) in model.NamedTrainableTensors())
        {
            var grad = tape[key].ToArray();
            // Probe the largest-magnitude entries (for the token table: the rows this window actually uses).
            foreach (int i in Enumerable.Range(0, grad.Length).OrderByDescending(j => Math.Abs(grad[j])).Take(3))
            {
                double orig = param[i];
                param[i] = orig + eps; double lp = model.ReferenceSampleLoss(windows[0], targets[0]);
                param[i] = orig - eps; double lm = model.ReferenceSampleLoss(windows[0], targets[0]);
                param[i] = orig;
                double fd = (lp - lm) / (2 * eps);
                probes++;
                Assert.True(Math.Abs(grad[i] - fd) <= 1e-5 * Math.Max(1, Math.Abs(fd)),
                    $"{key}[{i}]: tape {grad[i]:E6} vs finite difference {fd:E6}");
            }
        }
        _out.WriteLine($"{probes} probes agree with finite differences");
        Assert.True(probes >= 3 * 29);
    }

    [Fact]
    [Trait("category", "unit")]
    public void BatchGradient_EqualsSumOfSingleWindowGradients()
    {
        var model = Model();
        var (windows, targets) = Samples(8, 5, 99);
        var batched = model.ComputeBatchGradientsTape(windows, targets);
        var summed = new Dictionary<string, double[]>();
        for (int s = 0; s < windows.Count; s++)
            foreach (var kv in model.ComputeBatchGradientsTape(new[] { windows[s] }, new[] { targets[s] }))
            {
                var a = kv.Value.ToArray();
                if (summed.TryGetValue(kv.Key, out var acc)) for (int i = 0; i < a.Length; i++) acc[i] += a[i];
                else summed[kv.Key] = a;
            }
        foreach (var (key, expected) in summed)
        {
            var got = batched[key].ToArray();
            double maxAbs = expected.Select(Math.Abs).DefaultIfEmpty(0).Max();
            double maxErr = expected.Zip(got, (e, g) => Math.Abs(e - g)).DefaultIfEmpty(0).Max();
            Assert.True(maxErr <= 1e-10 * Math.Max(1, maxAbs), $"{key}: batched vs summed single-window max |error| {maxErr:E3}");
        }
    }
}

/// <summary>Batched Predict over many rows must return exactly what per-row PredictSingle returns.</summary>
public sealed class ChronosBatchedPredictTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(5)]
    [InlineData(12)]
    [Trait("category", "unit")]
    public void BatchedPredict_MatchesPerRowPredictSingle(int windowLength)
    {
        var model = new ChronosFoundationModel<double>(new ChronosOptions<double>
        {
            Seed = 9, ContextLength = 8, ForecastHorizon = 1, EmbeddingDim = 8, NumLayers = 2, NumHeads = 2,
            VocabularySize = 64, Epochs = 2, UseEarlyStopping = false,
        });
        var trainX = new Matrix<double>(80, 1);
        var trainY = new Vector<double>(80);
        for (int i = 0; i < 80; i++) { trainX[i, 0] = i; trainY[i] = Math.Sin(i * 0.3) + 0.02 * i; }
        model.Train(trainX, trainY);

        var rng = new Random(windowLength);
        var input = new Matrix<double>(23, windowLength);
        for (int i = 0; i < input.Rows; i++)
            for (int j = 0; j < windowLength; j++) input[i, j] = Math.Sin(0.3 * (i + j)) + 0.2 * rng.NextDouble();

        var batched = model.Predict(input);
        for (int i = 0; i < input.Rows; i++)
            Assert.Equal(model.PredictSingle(input.GetRow(i)), batched[i]);
    }
}
