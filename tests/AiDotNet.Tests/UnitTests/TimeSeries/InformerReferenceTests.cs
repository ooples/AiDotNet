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
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// The shared Informer network against the official Informer2020 model (tools/reference-data/informer_reference.py):
/// ProbSparse and full attention forecasts, the MSE loss, and the loss after one SGD step, from the reference's weights
/// and its sampled key indices.
/// </summary>
public class InformerReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "informer_reference.json";
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

    private static Tensor<double> Tensor(JsonElement f, string name, params int[] shape)
        => new(shape, new Vector<double>(Values(f.GetProperty(name))));

    private static (InformerNetwork<double> net, List<LayerBase<double>> layers) Network(JsonElement f, InformerAttention attention)
    {
        var c = f.GetProperty("config");
        int I(string n) => c.GetProperty(n).GetInt32();
        var configuration = new InformerConfiguration(I("channels"), I("channels"), I("channels"), I("d_model"), I("n_heads"),
            I("e_layers"), I("d_layers"), I("d_ff"), I("factor"), 0.0, Distil: true, Mix: true, attention, Gelu: true,
            I("time_features"), I("label_len"), I("pred_len"));
        var layers = new List<LayerBase<double>>();
        var net = new InformerNetwork<double>(AiDotNetEngine.Current, layers, configuration);
        var state = f.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var draws = new Queue<int[][]>(f.GetProperty("draws").EnumerateArray()
            .Select(d => d.EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray()));
        net.KeySampler = (queries, samples, keys) =>
        {
            var draw = draws.Dequeue();
            Assert.Equal(queries, draw.Length);
            Assert.Equal(samples, draw[0].Length);
            var sampled = new int[queries, samples];
            for (int i = 0; i < queries; i++)
                for (int s = 0; s < samples; s++) sampled[i, s] = draw[i][s];
            return sampled;
        };
        return (net, layers);
    }

    private static (Tensor<double> xEnc, Tensor<double> markEnc, Tensor<double> xDec, Tensor<double> markDec, Tensor<double> target) Inputs(JsonElement f)
    {
        var c = f.GetProperty("config");
        int ch = c.GetProperty("channels").GetInt32(), seq = c.GetProperty("seq_len").GetInt32(), label = c.GetProperty("label_len").GetInt32();
        int pred = c.GetProperty("pred_len").GetInt32(), marks = c.GetProperty("time_features").GetInt32();
        return (Tensor(f, "x_enc", 2, seq, ch), Tensor(f, "mark_enc", 2, seq, marks), Tensor(f, "x_dec", 2, label + pred, ch),
            Tensor(f, "mark_dec", 2, label + pred, marks), Tensor(f, "target", 2, pred, ch));
    }

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{i}: {expected[i]} vs {actual[i]}");
    }

    private static double Mse(Tensor<double> a, Tensor<double> b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
        return s / a.Length;
    }

    [Fact(Timeout = 120000)]
    public async Task ProbSparseForecast_Loss_AndOneSgdStep_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var (net, layers) = Network(f, InformerAttention.ProbSparse);
        var (xEnc, markEnc, xDec, markDec, target) = Inputs(f);
        var random = new Random(0);

        Tensor<double> forecast;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        IReadOnlyList<Tensor<double>> parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        using (var tape = new GradientTape<double>())
        {
            forecast = net.Forward(xEnc, markEnc, xDec, markDec, training: false, random);
            var diff = AiDotNetEngine.Current.TensorSubtract(forecast, target);
            var loss = AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(diff, diff), new[] { 0, 1, 2 }, false);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Values(f.GetProperty("forecast")), forecast, 1e-9);
        Assert.True(Math.Abs(f.GetProperty("loss").GetDouble() - Mse(forecast, target)) < 1e-10);

        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        var after = net.Forward(xEnc, markEnc, xDec, markDec, training: false, random);
        Assert.True(Math.Abs(f.GetProperty("loss_after_sgd").GetDouble() - Mse(after, target)) < 1e-9,
            $"{f.GetProperty("loss_after_sgd").GetDouble()} vs {Mse(after, target)}");
    }

    [Fact(Timeout = 120000)]
    public async Task FullAttentionForecast_MatchesTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var (net, _) = Network(f, InformerAttention.Full);
        var (xEnc, markEnc, xDec, markDec, _) = Inputs(f);
        var forecast = net.Forward(xEnc, markEnc, xDec, markDec, training: false, new Random(0));
        AssertClose(Values(f.GetProperty("full_forecast")), forecast, 1e-9);
    }
}
