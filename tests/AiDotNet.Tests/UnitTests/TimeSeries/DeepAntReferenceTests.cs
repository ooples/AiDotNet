using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries.AnomalyDetection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// DeepAnT's predictor against the reference module (tools/reference-data/deepant_reference.py): the prediction, the
/// mean absolute error and the loss after one SGD step from the reference's weights; and the model's training,
/// forecasting and anomaly detection.
/// </summary>
public class DeepAntReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "deepant_reference.json";
        string relative = Path.Combine("UnitTests", "TimeSeries", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static double Mae(Tensor<double> a, Tensor<double> b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += Math.Abs(a[i] - b[i]);
        return s / a.Length;
    }

    [Fact(Timeout = 120000)]
    public async Task Prediction_Loss_AndOneSgdStep_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var c = f.GetProperty("config");
        int window = c.GetProperty("seq_len").GetInt32(), horizon = c.GetProperty("p_w").GetInt32();
        var layers = new List<LayerBase<double>>();
        var net = new DeepAntNetwork<double>(AiDotNetEngine.Current, layers, new DeepAntConfiguration(window, horizon,
            c.GetProperty("filters").GetInt32(), c.GetProperty("kernel").GetInt32(), c.GetProperty("pool").GetInt32(),
            c.GetProperty("hidden").GetInt32(), 0.25));
        var state = f.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        // [3, 1, w] in the reference is [3, w, 1] here; with one channel the values are in the same order.
        var windows = new Tensor<double>(new[] { 3, window, 1 }, new Vector<double>(Values(f.GetProperty("windows"))));
        var targets = new Tensor<double>(new[] { 3, horizon }, new Vector<double>(Values(f.GetProperty("targets"))));
        var random = new Random(0);

        IReadOnlyList<Tensor<double>> parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Tensor<double> prediction;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            prediction = net.Forward(windows, training: false, random);
            var loss = new AiDotNet.LossFunctions.MeanAbsoluteErrorLoss<double>().ComputeTapeLoss(prediction, targets);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        var expected = Values(f.GetProperty("prediction"));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - prediction[i]) < 1e-12, $"{i}: {expected[i]} vs {prediction[i]}");
        Assert.True(Math.Abs(f.GetProperty("loss").GetDouble() - Mae(prediction, targets)) < 1e-12);

        Assert.Equal(8, parameters.Count);
        foreach (var p in parameters)
        {
            Assert.True(grads.TryGetValue(p, out var g), "every parameter has a gradient");
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g![i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        var after = net.Forward(windows, training: false, random);
        Assert.True(Math.Abs(f.GetProperty("loss_after_sgd").GetDouble() - Mae(after, targets)) < 1e-12,
            $"{f.GetProperty("loss_after_sgd").GetDouble()} vs {Mae(after, targets)}");
    }

    private static (Matrix<double> x, Vector<double> y) Series(int length, Func<int, double> value)
    {
        var x = new Matrix<double>(length, 1);
        var y = new Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            x[i, 0] = i;
            y[i] = value(i);
        }
        return (x, y);
    }

    private static double Seasonal(int t) => 10 + 3 * Math.Sin(2 * Math.PI * t / 12.0);

    private static DeepANTOptions<double> Small() => new()
    {
        WindowSize = 24, Epochs = 150, LearningRate = 0.05, Seed = 3,
    };

    [Fact(Timeout = 120000)]
    public async Task Training_FitsASeasonalSeries_AndForecastsItsNextPeriod()
    {
        await Task.Yield();
        var (x, y) = Series(240, Seasonal);
        var model = new DeepANT<double>(Small());
        model.Train(x, y);

        // Out of sample: the 12 steps after the series, recursively.
        var future = new Matrix<double>(12, 1);
        var forecast = model.Predict(future);
        double error = 0;
        for (int h = 0; h < 12; h++) error += Math.Abs(forecast[h] - Seasonal(240 + h));
        error /= 12;
        Assert.True(error < 0.75, $"mean absolute forecast error {error} (amplitude 3)");

        // The scores are normalized prediction errors: small on the series it learned.
        var scores = model.ComputeAnomalyScores(y);
        Assert.Equal(240 - 24, scores.Length);
        Assert.True(scores.Average() < 0.1, $"mean score {scores.Average()}");
    }

    [Fact(Timeout = 120000)]
    public async Task DetectAnomalies_FlagsASpike_AndNothingElse()
    {
        await Task.Yield();
        var (x, y) = Series(240, Seasonal);
        var model = new DeepANT<double>(Small());
        model.Train(x, y);
        var test = new Vector<double>(120);
        for (int i = 0; i < test.Length; i++) test[i] = Seasonal(240 + i);
        test[80] += 6;
        var flags = model.DetectAnomalies(test);
        Assert.True(flags[80 - 24], "the spike is flagged");
        Assert.Equal(1, flags.Count(b => b));
    }

    [Fact]
    public void Construction_RejectsAWindowTooShortForTheTwoBlocks()
    {
        // (6 − 2) / 2 = 2, then (2 − 2) / 2 = 0 steps left.
        var ex = Assert.Throws<ArgumentException>(() => new DeepANT<double>(new DeepANTOptions<double> { WindowSize = 6 }));
        Assert.Contains("window of 6", ex.Message);
    }

    [Fact(Timeout = 120000)]
    public async Task AMultiStepHorizon_TrainsAndPredictsTheNextValue()
    {
        await Task.Yield();
        var (x, y) = Series(120, Seasonal);
        var options = Small();
        options.ForecastHorizon = 3;
        var model = new DeepANT<double>(options);
        model.Train(x, y);
        var window = new Vector<double>(24);
        for (int t = 0; t < 24; t++) window[t] = y[96 + t];
        double next = model.PredictSingle(window);
        Assert.True(Math.Abs(next - Seasonal(120)) < 1.0, $"{next} vs {Seasonal(120)}");
    }
}
